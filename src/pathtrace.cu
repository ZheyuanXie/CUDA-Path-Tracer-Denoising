#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geom = NULL;
static Material * dev_material = NULL;
static Path * dev_path = NULL;

static glm::vec3 * dev_lightColor = NULL;	// one for each pixel, used for shading stage
static DiffusePhongShadingChunk * dev_diffuse = NULL;
static SpecularPhongShadingChunk * dev_specular = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_path, pixelcount * sizeof(Path));

	cudaMalloc(&dev_geom, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geom, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_material, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_material, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lightColor, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_lightColor, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_diffuse, pixelcount * sizeof(DiffusePhongShadingChunk));
	cudaMemset(dev_diffuse, 0, pixelcount * sizeof(DiffusePhongShadingChunk));

	cudaMalloc(&dev_specular, pixelcount * sizeof(SpecularPhongShadingChunk));
	cudaMemset(dev_specular, 0, pixelcount * sizeof(SpecularPhongShadingChunk));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_path);
	cudaFree(dev_geom);
	cudaFree(dev_material);

	cudaFree(dev_lightColor);
	cudaFree(dev_diffuse);
	cudaFree(dev_specular);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate Rays from camera through screen to the field
* which is the first generation of rays
*
* Antialiasing - num of rays per pixel
* motion blur - jitter scene position
* lens effect - jitter camera position
*/
__global__ void generateRayFromCamera(Camera cam, int iter, Path* paths, glm::vec3 * lightColor)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		Path & path = paths[index];

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		path.ray.origin = cam.position;

		// TODO: implement antialiasing by jittering the ray
		path.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		path.pixelIndex = index;
		path.color = glm::vec3(1.0f);
		path.terminated = false;

		// init lightColor to 1.0
		lightColor[index] = glm::vec3(1.0);
	}
}

__global__ void pathTraceOneBounce(
	int depth
	, int num_paths
	, glm::vec3 * lightColor
	, Path * paths
	, Geom * geoms
	, int geoms_size
	, Material * materials
	, DiffusePhongShadingChunk * diffuse
	, SpecularPhongShadingChunk * specular
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		Path path = paths[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, path.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, path.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more primitive types intersection test here

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}


		///////////////////////////////


		if (hit_geom_index == -1)
		{
			// The ray hits nothing, no need to shade the pixel that this ray contributes to
			path.terminated = true;
			
			// Set the 100% ray color of this pixel to 0x000000
			lightColor[path.pixelIndex] = glm::vec3(0.0);
		}
		else
		{
			//The ray hits something 
			Geom & geom = geoms[hit_geom_index];
			Material & material = materials[geom.materialid];
			
			if (material.emittance > EPSILON)
			{
				// we hit a light source
				path.terminated = true;

				// Set the 100% ray color of this pixel, will be used in the shading stage
				lightColor[path.pixelIndex] *= material.color * material.emittance;


				// -----------------
				// TODO: Delete the line below, this is to give you an intuition of path tracing
				//image[path.pixelIndex] += material.color * material.emittance;
			}
			else
			{
				// we hit a surface (not a light)
				path.terminated = false;
				
				// TODO: push a ShadingChunk to its list based on the material
				// which will be used in the shading stage
				
				// TODO: delete me, test only
				diffuse[path_index].activated = true;
				diffuse[path_index].materialId = geom.materialid;
				diffuse[path_index].pixelId = path.pixelIndex;


				// TODO: call scatterRay






				// ------------------
				// TODO: Delete the line below, this is to give you an intuition of path tracing
				//image[path.pixelIndex] += normal;
			}
		}

	}
}



__global__ void shadeDiffusePhongMaterial (
	int num_chunks
	, DiffusePhongShadingChunk * diffuse_chunk
	, glm::vec3 * lightColor
	, Material * materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_chunks)
	{
		DiffusePhongShadingChunk & chunk = diffuse_chunk[idx];

		if (chunk.activated)
		{
			lightColor[chunk.pixelId] *= materials[chunk.materialId].color;
		}
	}
}

__global__ void shadeDiffusePhongMaterial(
	int num_chunks
	, SpecularPhongShadingChunk * specular_chunk
	, glm::vec3 * lightColor
	, Material * materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_chunks)
	{
		SpecularPhongShadingChunk & chunk = specular_chunk[idx];

		if (chunk.activated)
		{
			// TODO
		}
	}
}



__global__ void finalGather(int x_resolution, int y_resolution, glm::vec3 * image, glm::vec3 * lightColor)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < x_resolution && y < y_resolution)
	{
		image[y * x_resolution + x] += lightColor[y * x_resolution + x];
	}
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray is a (ray, color) pair, where color starts as the
    //     multiplicative identity, white = (1, 1, 1).
    //   * For debugging, you can output your ray directions as colors.
    // * For each depth:
    //   * Compute one new (ray, color) pair along each path (using scatterRay).
    //     Note that many rays will terminate by hitting a light or hitting
    //     nothing at all. You'll have to decide how to represent your path rays
    //     and how you'll mark terminated rays.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //       surface.
    //     * You can debug your ray-scene intersections by displaying various
    //       values as colors, e.g., the first surface normal, the first bounced
    //       ray direction, the first unlit material color, etc.
    //   * Add all of the terminated rays' results into the appropriate pixels.
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    // * Finally, handle all of the paths that still haven't terminated.
    //   (Easy way is to make them black or background-colored.)

    // TODO: perform one iteration of path tracing

    //generateNoiseDeleteMe<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, dev_image);

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, dev_path, dev_lightColor);
	checkCUDAError("generate camera ray");

	int depth = 0;
	Path* dev_path_end = dev_path + pixelcount;
	int num_path = dev_path_end - dev_path;

	// --- Path Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	
	// TODO: write a loop to iterate your path tracing process
	// while (...) {

	// clean shading chunks
	cudaMemset(dev_diffuse, 0, pixelcount * sizeof(DiffusePhongShadingChunk));

	// tracing
	dim3 numblocksPathTracing = (num_path + blockSize1d - 1) / blockSize1d;
	pathTraceOneBounce << <numblocksPathTracing, blockSize1d >> > (
		depth
		, num_path
		, dev_lightColor
		, dev_path
		, dev_geom
		, hst_scene->geoms.size()
		, dev_material
		, dev_diffuse
		, dev_specular
		);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();
	depth++;


	// TODO:
	// --- Shading Stage ---
	// Given the shading chunks (one for each material)
	// Shade one material for one time by launching a kernel
	shadeDiffusePhongMaterial << <numblocksPathTracing, blockSize1d >> > (
		num_path
		, dev_diffuse
		, dev_lightColor
		, dev_material);


	
	//}		// brackets of while loop  


	
	finalGather << <blocksPerGrid2d, blockSize2d >> >(cam.resolution.x, cam.resolution.y, dev_image, dev_lightColor);


    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
