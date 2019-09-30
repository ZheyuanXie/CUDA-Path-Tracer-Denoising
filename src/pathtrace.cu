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
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static ShadeableIntersection * dev_intersections_cache = NULL;    // cache first iteration.
static Triangle * dev_triangles = NULL;                           // triangles

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

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

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

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
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

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
  , Triangle* triangles
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

#ifndef STREAM_COMPACTION
    if (pathSegment.remainingBounces <= 0) return;
#endif // !STREAM_COMPACTION

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
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *= u01(rng); // apply some noise because why not
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
    }
  }
}

// The implementation of the real shader
__global__ void shadeRealMaterial(
  int iter
  , int depth
  , int num_paths
  , ShadeableIntersection * shadeableIntersections
  , PathSegment * pathSegments
  , Material * materials
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment &pathSegment = pathSegments[idx];

#ifndef STREAM_COMPACTION
    if (pathSegment.remainingBounces <= 0) return;
#endif // !STREAM_COMPACTION

    if (intersection.t > 0.0f) { // if the intersection exists...

      // Set up the RNG
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      if (material.emittance > 0.0f) {  // Hit light (Terminate)
        pathSegment.color *= (materialColor * material.emittance);
        pathSegment.remainingBounces = -1;
      }
      else {  // Hit Material (Bounce)
        glm::vec3 intersect = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
        scatterRay(pathSegment, intersect, intersection.surfaceNormal, material, rng);
      }
    }
    else {    // No Intersection (Terminate)
      pathSegment.color = glm::vec3(0.0f);
      pathSegment.remainingBounces = -1;
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
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
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

	  generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	  checkCUDAError("generate camera ray");

	  int depth = 0;
	  PathSegment* dev_path_end = dev_paths + pixelcount;
	  int num_paths = dev_path_end - dev_paths;

	  // --- PathSegment Tracing Stage ---
	  // Shoot ray into scene, bounce between objects, push shading chunks
    time_start = std::chrono::high_resolution_clock::now();

    bool iterationComplete = false;
	  while (!iterationComplete) {

	      // clean shading chunks
	      cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	      // path trace
	      dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if defined(CACHE_FIRST_ITERATION)
        if (iter != 0 || (iter == 0 && depth == 0) || cam.antialiasing) {
          computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , dev_triangles
            , hst_scene->geoms.size()
            , dev_intersections
            );
          checkCUDAError("trace one bounce");
          if (iter == 0 && !cam.antialiasing) {      // bulid cache
            cudaMemcpy(dev_intersections_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
          }
        } else {                // use cache
          cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
#else
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
          depth
          , num_paths
          , dev_paths
          , dev_geoms
          , dev_triangles
          , hst_scene->geoms.size()
          , dev_intersections
          );
        checkCUDAError("trace one bounce");
#endif // CACHE_FIRST_ITERATION
        cudaDeviceSynchronize();
	      depth++;

	      // TODO:
	      // --- Shading Stage ---
	      // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

#ifdef SORT_BY_MATERIALS
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, material_id_comparator());
#endif // SORT_BY_MATERIALS

        // do shading
        shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
          iter,
          depth,
          num_paths,
          dev_intersections,
          dev_paths,
          dev_materials
        );
        cudaDeviceSynchronize();

#ifdef STREAM_COMPACTION
        dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, ray_continuation_condition());
        num_paths = dev_path_end - dev_paths;
        iterationComplete = (num_paths <= 0);
#else
        iterationComplete = (depth >= traceDepth);
#endif  // STREAM_COMPACTION
	  }

    time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duro = time_end - time_start;
    float prev_elapsed_time_cpu_milliseconds =
      static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());
    avg_time = (avg_time * (iter - 1) + prev_elapsed_time_cpu_milliseconds) / (iter);
    cout << "Iter:" << iter << ", Time:" << prev_elapsed_time_cpu_milliseconds << ", Avg Time:" << avg_time << endl;

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	  finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");

}
