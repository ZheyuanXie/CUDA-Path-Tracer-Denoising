#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <chrono>

#include "denoise.h"
#include "main.h"

////CPU////
static Scene * hst_scene = NULL;
static glm::mat4 view_matrix_prev;          // previous view projection matrix (CPU)
///////////

////GPU////
static glm::vec2 * dev_moment_history = NULL;
static glm::vec3 * dev_color_history = NULL;
static glm::vec3 * dev_color_acc = NULL;    // accumulated color
static glm::vec2 * dev_moment_acc = NULL;   // accumulated moment
static int * dev_history_length = NULL;    // history length before back projection
static GBufferTexel * dev_gbuffer_prev = NULL;
static float * dev_variance = NULL;
static glm::vec3 * dev_temp[2] = { NULL };  // ping-pong buffers used in a-tours wavelet transform
///////////


void denoiseInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_temp[0], pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_temp[0], 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_temp[1], pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_temp[1], 0, pixelcount * sizeof(glm::vec3));
    
    // allocate memory for history buffers
    cudaMalloc(&dev_history_length, pixelcount * sizeof(int));
    cudaMemset(dev_history_length, 0, pixelcount * sizeof(int));

    cudaMalloc(&dev_moment_history, pixelcount * sizeof(glm::vec2));
    cudaMemset(dev_moment_history, 0, pixelcount * sizeof(glm::vec2));
    cudaMalloc(&dev_moment_acc, pixelcount * sizeof(glm::vec2));
    cudaMemset(dev_moment_acc, 0, pixelcount * sizeof(glm::vec2));

    cudaMalloc(&dev_color_history, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_color_acc, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_gbuffer_prev, pixelcount * sizeof(GBufferTexel));

    cudaMalloc(&dev_variance, pixelcount * sizeof(float));
    cudaMemset(dev_variance, 0, pixelcount * sizeof(float));
}

void denoiseFree() {
    cudaFree(dev_temp[0]);
    cudaFree(dev_temp[1]);
    cudaFree(dev_moment_history);
    cudaFree(dev_color_history);
    cudaFree(dev_gbuffer_prev);
    cudaFree(dev_variance);
}

__global__ void initializeMoment(glm::vec2 res, glm::vec2 * dev_moment_history) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < res.x && y < res.y) {
        int p = x + y * res.x;
    }
}

// A-Trous Filter
__global__ void ATrousFilter(glm::vec3 * dev_input, glm::vec3 * dev_output, GBufferTexel * gBuffer, glm::ivec2 res, int level,
                             float variance, float sigma_n, float sigma_x)
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
        float weights_squared = 0;
        glm::vec3 color = glm::vec3(0.0f);
        //float variance = 0.0f;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                int xq = x + step * i;
                int yq = y + step * j;
                int q = xq + yq * res.x;
                if (xq >= 0 && xq < res.x && yq >= 0 && yq < res.y) {
                    int k = (2 + i) + (2 + j) * 5;

                    float wrt = 1.0f, wn = 1.0f, wx = 1.0f;

                    float lp = 0.2126 * dev_input[p].x + 0.7152 * dev_input[p].y + 0.0722 * dev_input[p].z;
                    float lq = 0.2126 * dev_input[q].x + 0.7152 * dev_input[q].y + 0.0722 * dev_input[q].z;

                    wrt = exp(-abs(lp - lq) / (variance + 1e-6));
                    wn = exp(-distance(gBuffer[p].normal, gBuffer[q].normal) / sigma_n);
                    wx = exp(-distance(gBuffer[p].position, gBuffer[q].position) / sigma_x);

                    float weight = h[k] * wrt * wn * wx;
                    color += (dev_input[q] * weight);
                    //variance += (dev_variance[q] * weight * weight);
                    weights += weight;
                    weights_squared += weight * weight;
                }
            }
        }

        dev_output[p] = color / (float)weights;
        //dev_variance[p] = variance / (float)weights_squared;
    }
}

__device__ bool isReprjValid(glm::ivec2 res, glm::vec2 curr_coord, glm::vec2 prev_coord, GBufferTexel * curr_gbuffer, GBufferTexel * prev_gbuffer) {
    int p = curr_coord.x + curr_coord.y * res.x;
    int q = prev_coord.x + prev_coord.y * res.x;
    // reject if the pixel is outside the screen
    if (prev_coord.x < 0 || prev_coord.x >= res.x || prev_coord.y < 0 || prev_coord.y >= res.y) return false;
    // reject if the pixel is a different geometry
    if (prev_gbuffer[q].geomId != curr_gbuffer[p].geomId) return false;
    // reject if the normal deviation is not acceptable
    if (distance(prev_gbuffer[q].normal, curr_gbuffer[p].normal) > 1.0f) return false;
    return true;
}

// TODO: back projection
__global__ void BackProjection(float * variacne_out, int * history_length, glm::vec2 * moment_history, glm::vec3 * color_history, glm::vec2 * moment_acc, glm::vec3 * color_acc, 
                               glm::vec3 * current_color, GBufferTexel * current_gbuffer, GBufferTexel * prev_gbuffer, glm::mat4 prev_VP, glm::ivec2 res,
                               float color_alpha_min, float moment_alpha_min)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < res.x && y < res.y) {
        int p = x + y * res.x;
        int N = history_length[p];
        glm::vec3 sample = current_color[p];
        float luminance = 0.2126 * sample.x + 0.7152 * sample.y + 0.0722 * sample.z;

        if (N > 0) {
            /////////////
            // Calculate NDC coordinates in previous frame (TODO: check correctness)
            glm::vec3 current_position = current_gbuffer[p].position;
            glm::vec4 viewspace_position = prev_VP * glm::vec4(current_position, 1.0f);
            float clipx = viewspace_position.x / viewspace_position.z /** tanf(PI / 4)*/;
            float clipy = viewspace_position.y / viewspace_position.z /** tanf(PI / 4)*/;
            float ndcx = clipx * 0.5 + 0.5;
            float ndcy = clipy * 0.5 + 0.5;
            float prevx = ndcx * res.x - 0.5;
            float prevy = ndcy * res.y - 0.5;
            /////////////
            
            bool v[4];
            float floorx = floor(prevx);
            float floory = floor(prevy);
            float fracx = prevx - floorx;
            float fracy = prevy - floory;
            glm::ivec2 offset[4] = { glm::ivec2(0,0), glm::ivec2(1,0), glm::ivec2(0,1), glm::ivec2(1,1) };
            
            bool valid = false;
            for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++) {
                glm::ivec2 loc = glm::ivec2(floorx, floory) + offset[sampleIdx];
                v[sampleIdx] = isReprjValid(res, glm::ivec2(x, y), loc, current_gbuffer, prev_gbuffer);
                valid = valid || v[sampleIdx];
            }

            float sumw = 0.0f;

            float w[4] = {
                (1 - fracx) * (1 - fracy),
                fracx * (1 - fracy),
                (1 - fracx) * fracy,
                fracx * fracy
            };

            if (valid) {
                glm::vec3 prevColor = glm::vec3(0.0f);
                glm::vec2 prevMoments = glm::vec2(0.0f);
                float prevHistoryLength = 0;
                for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++) {
                    glm::ivec2 loc = glm::ivec2(floorx, floory) + offset[sampleIdx];
                    int locq = loc.x + loc.y * res.x;
                    if (v[sampleIdx]) {
                        prevColor += w[sampleIdx] * color_history[locq];
                        prevMoments += w[sampleIdx] * moment_history[locq];
                        prevHistoryLength += w[sampleIdx] * history_length[locq];
                        sumw += w[sampleIdx];
                    }
                }

                valid = (sumw >= 0.01);
                prevColor = valid ? prevColor / sumw : glm::vec3(0.0f);
                prevMoments = valid ? prevMoments / sumw : glm::vec2(0.0f);
                prevHistoryLength = valid ? prevHistoryLength / sumw : 0.0f;

                // calculate alpha values that controls fade
                float color_alpha = max(1.0f / (float)(N + 1), color_alpha_min);
                float moment_alpha = max(1.0f / (float)(N + 1), moment_alpha_min);

                // incresase history length
                history_length[p] = (int)prevHistoryLength + 1;

                // color accumulation
                color_acc[p] = current_color[p] * color_alpha + prevColor * (1.0f - color_alpha);
                //color_history[p] = color_history[p] * (float)(N) / (float)(N + 1) + current_color[p] / (float)(N + 1);

                // moment accumulation
                float first_moment = moment_alpha * prevMoments.x + (1.0f - moment_alpha) * luminance;
                float second_moment = moment_alpha * prevMoments.y + (1.0f - moment_alpha) * luminance * luminance;
                moment_acc[p] = glm::vec2(first_moment, second_moment);

                // calculate variance from moments
                float variance = second_moment - first_moment * first_moment;
                variacne_out[p] = variance > 0.0f ? variance : 0.0f;
                return;
            }
        }

        // If there's no history
        history_length[p] = 1;
        color_acc[p] = current_color[p];
        moment_acc[p] = glm::vec2(luminance, luminance * luminance);
        variacne_out[p] = 10.0f;
    }
}

glm::mat4 GetViewMatrix(const Camera& cam) {
    //////////////////
    glm::mat4 viewMatrix(glm::vec4(-cam.right, 0), glm::vec4(-cam.up, 0), glm::vec4(cam.view, 0), glm::vec4(cam.position, 1));
    //////////////////
    return viewMatrix;
}

void denoise(int iter, glm::vec3 * input, glm::vec3 * output, GBufferTexel * gbuffer) {
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D blocks
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    /* Estimate Variance */
    
    if (ui_accumulate) BackProjection<<<blocksPerGrid2d, blockSize2d>>>(dev_variance, dev_history_length, dev_moment_history, dev_color_history,
                                                                        dev_moment_acc, dev_color_acc, input, gbuffer, dev_gbuffer_prev, view_matrix_prev, cam.resolution,
                                                                        ui_color_alpha, ui_moment_alpha);
    view_matrix_prev = GetViewMatrix(cam);
    cudaMemcpy(dev_gbuffer_prev, gbuffer, sizeof(GBufferTexel) * pixelcount, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_moment_history, dev_moment_acc, sizeof(glm::vec2) * pixelcount, cudaMemcpyDeviceToDevice);

    /* Apply A-Tours filter */
    if (ui_history_level == 0) cudaMemcpy(dev_color_history, dev_color_acc, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    for (int level = 1; level <= ui_atrous_nlevel; level++) {
        glm::vec3* src = (level == 1) ? dev_color_acc : dev_temp[level % 2];
        glm::vec3* dst = (level == ui_atrous_nlevel) ? output : dev_temp[(level + 1) % 2];
        ATrousFilter <<<blocksPerGrid2d, blockSize2d>>>(src, dst, gbuffer, cam.resolution, level, ui_variance, 1.f, 1.f);
        if (level == ui_history_level) cudaMemcpy(dev_color_history, dst, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
}
