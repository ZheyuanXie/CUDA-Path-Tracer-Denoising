#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <chrono>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "main.h"
#include "denoise.h"

using time_point_t = std::chrono::high_resolution_clock::time_point;
time_point_t time_start;
time_point_t time_end;
float avg_time = 0;

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

struct ray_continuation_condition {
  __host__ __device__ bool operator()(const PathSegment& s) {
    return s.remainingBounces > 0;
  }
};

struct material_id_comparator {
  __host__ __device__ bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2) {
    return s1.materialId < s2.materialId;
  }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int frame, int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | (frame << 13) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes two images to the OpenGL PBO directly.
__global__ void sendTwoImagesToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* leftImage, glm::vec3* rightImage) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        int leftIndex = x + (y * resolution.x * 2);
        int rightIndex = x + (y * resolution.x * 2) + resolution.x;
        
        glm::vec3 pix;
        glm::ivec3 color;

        // write to left (path traced) image pixel locations.
        pix = leftImage[index];
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);
        pbo[leftIndex].w = 0;
        pbo[leftIndex].x = color.x;
        pbo[leftIndex].y = color.y;
        pbo[leftIndex].z = color.z;

        // write to right (denoised) image pixel locations.
        pix = rightImage[index];
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);
        pbo[rightIndex].w = 0;
        pbo[rightIndex].x = color.x;
        pbo[rightIndex].y = color.y;
        pbo[rightIndex].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;

static ShadeableIntersection * dev_intersections_cache = NULL;    // cache first iteration.
static Triangle * dev_triangles = NULL;                           // triangles
static GBufferTexel * dev_gbuffer = NULL;                         // G-buffer for normal and depth
static glm::vec3 * dev_denoised_image = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    
    cudaMalloc(&dev_gbuffer, pixelcount * sizeof(GBufferTexel));

    cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);

    // TODO: clean up any extra device memory you created
    cudaFree(dev_intersections_cache);
    cudaFree(dev_triangles);
    cudaFree(dev_gbuffer);
    cudaFree(dev_denoised_image);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];
    
    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

    thrust::default_random_engine rng = makeSeededRandomEngine(0, iter, index, 0);
    thrust::uniform_real_distribution<float> u01(0, 1);

    // motion blur
    thrust::normal_distribution<float> n01(0, 1);
    float t = abs(n01(rng));
    glm::vec3 view = cam.view * (1 - t) + (cam.view + cam.motion) * t;

    if (cam.antialiasing) {
      segment.ray.direction = glm::normalize(view
        - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
      );
    } else {
      segment.ray.direction = glm::normalize(view
        - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
      );
    }

    if (cam.depth_of_field) {
      // sample point on lens
      float r = u01(rng) * cam.lens_radius;
      float theta = u01(rng) * 2 * PI;
      glm::vec3 p_lens(r * cos(theta), r * sin(theta), 0.0f);

      // compute point on plane of focus
      float ft = cam.focal_distance / glm::abs(segment.ray.direction.z);
      glm::vec3 p_focus = segment.ray.origin + ft * segment.ray.direction;

      // update ray for effect of lens
      segment.ray.origin += p_lens;
      segment.ray.direction = glm::normalize(p_focus - segment.ray.origin);
    }

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in shadeRealMaterial
__global__ void computeIntersections(int depth, int num_paths, PathSegment * pathSegments, Geom * geoms, Triangle* triangles,
                                     int geoms_size, ShadeableIntersection * intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

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
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, triangles, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
            intersections[path_index].geomId = -1;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
            intersections[path_index].geomId = hit_geom_index;
		}
	}
}

// The implementation of the real shader
__global__ void shadeRealMaterial(int iter, int depth, int frame, int num_paths, 
    ShadeableIntersection * shadeableIntersections, PathSegment * pathSegments, Material * materials)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment &pathSegment = pathSegments[idx];

    if (pathSegment.remainingBounces < 0) return;
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      thrust::default_random_engine rng = makeSeededRandomEngine(frame + 100, iter, idx, depth);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      if (material.emittance > 0.0f) {  // Hit light (Terminate)
        if (pathSegment.directLight) {
            pathSegment.color *= materialColor * 1.0f;
            pathSegment.remainingBounces = -1;
        }
        else {
            pathSegment.color *= (materialColor * material.emittance);
            pathSegment.remainingBounces = -1;
        }
      }
      else {  // Hit Material (Bounce)
        glm::vec3 intersect = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
        scatterRay(pathSegment, intersect, intersection.surfaceNormal, material, rng);
        if (pathSegments[idx].remainingBounces <= 0) {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
      }
    }
    else {    // No Intersection (Terminate)
      pathSegment.color = glm::vec3(0.0f);
      pathSegment.remainingBounces = -1;
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths, int iter)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
        image[index] = image[index] * (float)(iter - 1) / (float)iter + iterationPath.color / (float)iter;
	}
}

// Populate G-buffer
__global__ void populteGBuffer(int nPaths, GBufferTexel * gBuffer,
    ShadeableIntersection * shadeableIntersections, PathSegment * pathSegments, Material * materials)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths) {
        Ray ray = pathSegments[index].ray;
        ShadeableIntersection intersection = shadeableIntersections[index];
        gBuffer[index].position = ray.origin + intersection.t * ray.direction;
        gBuffer[index].normal = intersection.surfaceNormal;
        gBuffer[index].geomId = intersection.geomId;
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
    // start ray tracing timer
    time_start = std::chrono::high_resolution_clock::now();

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    bool iterationComplete = false;
    while (!iterationComplete) {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
          
        // path trace
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        if (iter != 1 || (iter == 1 && depth == 0) || cam.antialiasing) {
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (depth, num_paths, dev_paths, dev_geoms,
                                                                                dev_triangles, hst_scene->geoms.size(), dev_intersections);
            checkCUDAError("compute intersections");
            if (iter == 1 && !cam.antialiasing) {      // bulid cache
            cudaMemcpy(dev_intersections_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
        } else {                                       // use cache
            cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }

        if (iter == 1 && depth == 0) {
            populteGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_gbuffer, dev_intersections, dev_paths, dev_materials);
            checkCUDAError("populate G-buffer");
        }
        
        // increase depth by 1
        depth++;

        // do shading
        shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (iter, depth, frame, num_paths, 
                                                                         dev_intersections, dev_paths, dev_materials);
        checkCUDAError("shade material");

        // termination condition
        iterationComplete = (depth > traceDepth);
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths, iter);
    checkCUDAError("final gather");

    // conclude ray tracing timer
    time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duro = time_end - time_start;
    float prev_elapsed_time_cpu_milliseconds =
      static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());
    avg_time = (avg_time * (iter - 1) + prev_elapsed_time_cpu_milliseconds) / (iter);
    cout << "Iter:" << iter << ", Time:" << prev_elapsed_time_cpu_milliseconds << ", Avg Time:" << avg_time << endl;
    ///////////////////////////////////////////////////////////////////////////

    // Run denoiser!
    denoise(iter, dev_image, dev_denoised_image, dev_gbuffer);

    // Send results to OpenGL buffer for rendering
    sendTwoImagesToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image, dev_denoised_image);
    checkCUDAError("send images to PBO");

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}
