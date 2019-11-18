#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <chrono>

#include "denoise.h"

static Scene * hst_scene = NULL;
static glm::vec3 * dev_temp1 = NULL;
static glm::vec3 * dev_temp2 = NULL;

void denoiseInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_temp1, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_temp1, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_temp2, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_temp2, 0, pixelcount * sizeof(glm::vec3));
}

void denoiseFree() {
    cudaFree(dev_temp1);
    cudaFree(dev_temp2);
}

// A simple filter
__global__ void atrousFilter(int nPaths, glm::vec3 * input, glm::vec3 * output, GBufferTexel * gBuffer, glm::ivec2 res, int level,
                             float sigma_rt, float sigma_n, float sigma_x)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // a-trous kernel
    float h[25] = { 1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
                    1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
                    3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0, 3.0 / 32.0, 3.0 / 128.0,
                    1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
                    1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0 };

    if (x < res.x && y < res.y) {
        int p = x + y * res.x;
        int step = 1 << level;
        float weights = 0;
        glm::vec3 color = glm::vec3(0.0f);
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                int xq = x + step * i;
                int yq = y + step * j;
                int q = xq + yq * res.x;
                if (xq >= 0 && xq < res.x && yq >= 0 && yq < res.y) {
                    int k = (2 + i) + (2 + j) * 5;

                    float wrt = 1.0f, wn = 1.0f, wx = 1.0f;

                    wrt = exp(-distance(input[p], input[q]) / sigma_rt);
                    wn = exp(-distance(gBuffer[p].normal, gBuffer[q].normal) / sigma_n);
                    wx = exp(-distance(gBuffer[p].position, gBuffer[q].position) / sigma_x);

                    float weight = h[k] * wrt * wn * wx;
                    color += (input[q] * weight);
                    weights += weight;
                }
            }
        }
        output[p] = color / (float)weights;
    }
}

void denoise(int iter, glm::vec3 * input, glm::vec3 * output, GBufferTexel * gbuffer) {
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D blocks
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    float sigma_rt = 50.f / (float)iter;

    atrousFilter<<<blocksPerGrid2d, blockSize2d>>>(pixelcount, input, dev_temp1, gbuffer, cam.resolution, 1, sigma_rt, 1.f, 1.f);
    atrousFilter<<<blocksPerGrid2d, blockSize2d>>>(pixelcount, dev_temp1, dev_temp2, gbuffer, cam.resolution, 2, sigma_rt / 2, 1.f, 1.f);
    atrousFilter<<<blocksPerGrid2d, blockSize2d>>>(pixelcount, dev_temp2, dev_temp1, gbuffer, cam.resolution, 3, sigma_rt / 4, 1.f, 1.f);
    atrousFilter<<<blocksPerGrid2d, blockSize2d>>>(pixelcount, dev_temp1, dev_temp2, gbuffer, cam.resolution, 4, sigma_rt / 8, 1.f, 1.f);
    atrousFilter<<<blocksPerGrid2d, blockSize2d>>>(pixelcount, dev_temp2, output, gbuffer, cam.resolution, 5, sigma_rt / 16, 1.f, 1.f);

    cudaDeviceSynchronize();
}
