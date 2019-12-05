#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

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
thrust::default_random_engine makeSeededRandomEngine(int frame, int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 27) | (frame << 13) | iter) ^ utilhash(index);
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

    cudaFree(dev_triangles);
    cudaFree(dev_gbuffer);
    cudaFree(dev_denoised_image);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);

        // initial ray
		PathSegment & segment = pathSegments[index];
        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f);
        segment.ray.direction = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)x - (float)(cam.resolution.x * 0.5f - 0.5f))
        - cam.up * cam.pixelLength.y * ((float)y - (float)(cam.resolution.y * 0.5f - 0.5f))
        );
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
	}
}

__host__ __device__
bool computeIntersection(Ray& ray, ShadeableIntersection& intersection, Geom * geoms, int geoms_size, Triangle* triangles) {
    // closest hit
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    bool outside;

    float t;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    bool tmp_outside;

    for (int i = 0; i < geoms_size; i++)
    {
        Geom & geom = geoms[i];
        if (geom.type == CUBE) t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_outside);
        else if (geom.type == SPHERE) t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_outside);
        else if (geom.type == MESH) t = meshIntersectionTest(geom, triangles, ray, tmp_intersect, tmp_normal, tmp_outside);

        // update closest hit
        if (t > 0.0f && t < t_min) {
            t_min = t;
            hit_geom_index = i;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
            outside = tmp_outside;
        }
    }

    if (hit_geom_index == -1) {
        // The ray hits nothing
        intersection.t = -1.0f;
        intersection.geomId = -1;
        return false;
    } else {
        //The ray hits something
        intersection.t = t_min;
        intersection.materialId = geoms[hit_geom_index].materialid;
        intersection.surfaceNormal = normal;
        intersection.geomId = hit_geom_index;
        intersection.outside = outside;
        return true;
    }
}

// TODO
__host__ __device__
void computeShadowRay(Ray& shadowRay, glm::vec3 originPos, Geom& light, unsigned int& seed) {
    // random sample in a unit circle prependiculer to the direction to light
    glm::vec3 directionToCenter = glm::normalize(light.translation - originPos);
    glm::quat rot = glm::rotation(glm::vec3(0.0f, 0.0f, 1.0f), directionToCenter);
    float theta = 2 * PI * nextRand(seed);
    glm::vec3 sampleDirection = glm::rotate(rot, glm::vec3(cosf(theta), sinf(theta), 0.0f));

    float lightRadius = 0.5f;
    shadowRay.origin = originPos;
    shadowRay.direction = glm::normalize(light.translation + sampleDirection * lightRadius - originPos);
}

// do ray tracing kernel
__global__ void rt(int frame, int num_paths, int max_depth,
    PathSegment * pathSegments, ShadeableIntersection * intersections, 
    Geom * geoms, int geoms_size, Triangle* triangles, Material * materials, GBufferTexel * gbuffer, glm::vec3 * image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& segment = pathSegments[idx];
        ShadeableIntersection& intersection = intersections[idx];
        glm::vec3 accumulatedColor(0.0f);
        for (int depth = 1; depth <= max_depth; depth++) {
            bool hit = computeIntersection(segment.ray, intersection, geoms, geoms_size, triangles);

            // g-buffer
            if (depth == 1) {
                gbuffer[idx].position = segment.ray.origin + intersection.t * segment.ray.direction;
                gbuffer[idx].normal = intersection.surfaceNormal;
                gbuffer[idx].geomId = intersection.geomId;
            }

            if (!hit) break;

            unsigned int seed = initRand(idx, frame * depth, 16);
            Material material = materials[intersection.materialId];
            if (material.emittance > 0.0f) {  // Hit light (terminate ray)
                accumulatedColor += segment.color * material.color * material.emittance;
                break;
            }
            else {                            // Hit material (scatter ray)
                glm::vec3 intersectionPos = segment.ray.origin + intersection.t * segment.ray.direction;
                glm::vec3 intersectionNormal = intersection.surfaceNormal;

                // color mask
                segment.color *= material.color;
                glm::clamp(segment.color, glm::vec3(0.0f), glm::vec3(1.0f));

                // trace shadow ray
                if (true) {
                    Ray shadowRay;
                    float pdf;
                    computeShadowRay(shadowRay, intersectionPos + 1e-4f * intersectionNormal, geoms[0], seed);
                    ShadeableIntersection shadowRayIntersection;
                    bool shadowRayHit = computeIntersection(shadowRay, shadowRayIntersection, geoms, geoms_size, triangles);
                    if (shadowRayHit) {
                        Material shadowRayMaterial = materials[shadowRayIntersection.materialId];
                        if (shadowRayMaterial.emittance > 0.0f) {
                            glm::vec3 shadowRayIntersectionPos = shadowRay.origin + shadowRay.direction * shadowRayIntersection.t;
                            float diffuse = glm::max(0.0f, glm::dot(shadowRay.direction, intersectionNormal));
                            float shadowIntensity = 1.f / (shadowRayIntersection.t * shadowRayIntersection.t);  // TODO
                            accumulatedColor += segment.color * material.color
                                                * shadowRayMaterial.emittance * shadowRayMaterial.color
                                                * shadowIntensity * diffuse;
                        }
                    }
                }

                // bounce ray
                scatterRay(segment, intersectionPos, intersectionNormal, material, seed);
            }
        }
        image[segment.pixelIndex] = glm::clamp(accumulatedColor, glm::vec3(0.0f), glm::vec3(1.0f));
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;
    dim3 blocksPerGrid1d = (pixelcount + blockSize1d - 1) / blockSize1d;

    ///////////////////////////////////////////////////////////////////////////

    // Generate camera rays
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d >>>(cam, iter, ui_tracedepth, dev_paths);
    checkCUDAError("generate camera ray");
    
    // Do actual ray tracing
    rt<<<blocksPerGrid1d, blockSize1d>>>(frame, pixelcount, ui_tracedepth,
        dev_paths, dev_intersections,
        dev_geoms, hst_scene->geoms.size(),
        dev_triangles, dev_materials, dev_gbuffer, dev_image);
    checkCUDAError("ray tracing");

    // Run denoiser!
    if (ui_denoise_enable) {
        denoise(iter, dev_image, dev_denoised_image, dev_gbuffer);
    } else {
        cudaMemcpy(dev_denoised_image, dev_image, sizeof(glm::vec3) * pixelcount, cudaMemcpyDeviceToDevice);
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendTwoImagesToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image, dev_denoised_image);
    checkCUDAError("send images to PBO");

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}
