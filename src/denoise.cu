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
static int * dev_history_length = NULL;             // history length before back projection
static int * dev_history_length_update = NULL;      // history length after back projection
static GBufferTexel * dev_gbuffer_prev = NULL;
static float * dev_variance = NULL;
static glm::vec3 * dev_temp[2] = { NULL };  // ping-pong buffers used in a-tours wavelet transform
///////////


void denoiseInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // temporary ping-pong buffers
    cudaMalloc(&dev_temp[0], pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_temp[1], pixelcount * sizeof(glm::vec3));
    
    // history length
    cudaMalloc(&dev_history_length, pixelcount * sizeof(int));
    cudaMemset(dev_history_length, 0, pixelcount * sizeof(int));
    cudaMalloc(&dev_history_length_update, pixelcount * sizeof(int));

    // moment history and moment accumulation (history + current -> accumulation)
    cudaMalloc(&dev_moment_history, pixelcount * sizeof(glm::vec2));
    cudaMemset(dev_moment_history, 0, pixelcount * sizeof(glm::vec2));
    cudaMalloc(&dev_moment_acc, pixelcount * sizeof(glm::vec2));
    cudaMemset(dev_moment_acc, 0, pixelcount * sizeof(glm::vec2));

    // color history and color accumulation (history + current -> accumulation)
    cudaMalloc(&dev_color_history, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_color_acc, pixelcount * sizeof(glm::vec3));

    // store gbuffer from the previous frame
    cudaMalloc(&dev_gbuffer_prev, pixelcount * sizeof(GBufferTexel));

    // per pixel variance
    cudaMalloc(&dev_variance, pixelcount * sizeof(float));
    cudaMemset(dev_variance, 0, pixelcount * sizeof(float));
}

void denoiseFree() {
    cudaFree(dev_temp[0]);
    cudaFree(dev_temp[1]);
    cudaFree(dev_history_length);
    cudaFree(dev_history_length_update);
    cudaFree(dev_moment_acc);
    cudaFree(dev_color_acc);
    cudaFree(dev_moment_history);
    cudaFree(dev_color_history);
    cudaFree(dev_gbuffer_prev);
    cudaFree(dev_variance);
}

// A-Trous Filter
__global__ void ATrousFilter(glm::vec3 * colorin, glm::vec3 * colorout, float * variance,
                             GBufferTexel * gBuffer, glm::ivec2 res, int level, bool is_last,
                             float sigma_c, float sigma_n, float sigma_x, bool blur_variance, bool addcolor)
{
    // 5x5 A-Trous kernel
    float h[25] = { 1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
                    1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
                    3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0, 3.0 / 32.0, 3.0 / 128.0,
                    1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
                    1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0 };
    
    // 3x3 Gaussian kernel
    float gaussian[9] = { 1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
                          1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
                          1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0 };

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < res.x && y < res.y) {
        int p = x + y * res.x;
        int step = 1 << level;
        
        float var;
        // perform 3x3 gaussian blur on variance
        if (blur_variance) {
            float sum = 0.0f;
            float sumw = 0.0f;
            glm::ivec2 g[9] = { glm::ivec2(-1, -1), glm::ivec2(0, -1), glm::ivec2(1, -1),
                               glm::ivec2(-1, 0),  glm::ivec2(0, 0),  glm::ivec2(1, 0),
                               glm::ivec2(-1, 1),  glm::ivec2(0, 1),  glm::ivec2(1, 1) };
            for (int sampleIdx = 0; sampleIdx < 9; sampleIdx++) {
                glm::ivec2 loc = glm::ivec2(x, y) + g[sampleIdx];
                if (loc.x >= 0 && loc.y >= 0 && loc.x < res.x && loc.y < res.y) {
                    sum += gaussian[sampleIdx] * variance[loc.x + loc.y * res.x];
                    sumw += gaussian[sampleIdx];
                }
            }
            var = max(sum / sumw, 0.0f);
        } else {
            var = max(variance[p], 0.0f);
        }
        
        // Load pixel p data
        float lp = 0.2126 * colorin[p].x + 0.7152 * colorin[p].y + 0.0722 * colorin[p].z;
        glm::vec3 pp = gBuffer[p].position;
        glm::vec3 np = gBuffer[p].normal;

        glm::vec3 color_sum = glm::vec3(0.0f);
        float variance_sum = 0.0f;
        float weights_sum = 0;
        float weights_squared_sum = 0;

        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                int xq = x + step * i;
                int yq = y + step * j;
                if (xq >= 0 && xq < res.x && yq >= 0 && yq < res.y) {                    
                    int q = xq + yq * res.x;

                    // Load pixel q data
                    float lq = 0.2126 * colorin[q].x + 0.7152 * colorin[q].y + 0.0722 * colorin[q].z;
                    glm::vec3 pq = gBuffer[q].position;
                    glm::vec3 nq = gBuffer[q].normal;
                    
                    // Edge-stopping weights
                    float wl = expf(-glm::distance(lp, lq) / (sqrt(var) * sigma_c + 1e-6));
                    float wn = min(1.0f, expf(-glm::distance(np, nq) / (sigma_n + 1e-6)));
                    float wx = min(1.0f, expf(-glm::distance(pp, pq) / (sigma_x + 1e-6)));

                    // filter weights
                    int k = (2 + i) + (2 + j) * 5;
                    float weight = h[k] * wl * wn * wx;
                    weights_sum += weight;
                    weights_squared_sum += weight * weight;
                    color_sum += (colorin[q] * weight);
                    variance_sum += (variance[q] * weight * weight);
                }
            }
        }

        // update color and variance
        if (weights_sum > 10e-6) {
            colorout[p] = color_sum / weights_sum;
            variance[p] = variance_sum / weights_squared_sum;
        } else {
            colorout[p] = colorin[p];
        }

        if (is_last && addcolor) {
            colorout[p] *= gBuffer[p].albedo * gBuffer[p].ialbedo;
        }
    }
}

__device__ bool isReprjValid(glm::ivec2 res, glm::vec2 curr_coord, glm::vec2 prev_coord, GBufferTexel * curr_gbuffer, GBufferTexel * prev_gbuffer) {
    int p = curr_coord.x + curr_coord.y * res.x;
    int q = prev_coord.x + prev_coord.y * res.x;
    // reject if the pixel is outside the screen
    if (prev_coord.x < 0 || prev_coord.x >= res.x || prev_coord.y < 0 || prev_coord.y >= res.y) return false;
    // reject if the pixel is a different geometry
    if (prev_gbuffer[q].geomId == -1 || prev_gbuffer[q].geomId != curr_gbuffer[p].geomId) return false;
    // reject if the normal deviation is not acceptable
    if (distance(prev_gbuffer[q].normal, curr_gbuffer[p].normal) > 1e-1f) return false;
    return true;
}

// TODO: back projection
__global__ void BackProjection(float * variacne_out, int * history_length, int * history_length_update, glm::vec2 * moment_history, glm::vec3 * color_history, glm::vec2 * moment_acc, glm::vec3 * color_acc,
                               glm::vec3 * current_color, GBufferTexel * current_gbuffer, GBufferTexel * prev_gbuffer, glm::mat4 prev_viewmat, glm::ivec2 res,
                               float color_alpha_min, float moment_alpha_min)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < res.x && y < res.y) {
        int p = x + y * res.x;
        int N = history_length[p];
        glm::vec3 sample = current_color[p];
        float luminance = 0.2126 * sample.x + 0.7152 * sample.y + 0.0722 * sample.z;

        if (N > 0 && current_gbuffer[p].geomId != -1) {
            /////////////
            // Calculate NDC coordinates in previous frame (TODO: check correctness)
            glm::vec4 viewspace_position = prev_viewmat * glm::vec4(current_gbuffer[p].position, 1.0f);
            float clipx = viewspace_position.x / viewspace_position.z /** tanf(PI / 4)*/;
            float clipy = viewspace_position.y / viewspace_position.z /** tanf(PI / 4)*/;
            float ndcx = -clipx * 0.5f + 0.5f;
            float ndcy = -clipy * 0.5f + 0.5f;
            float prevx = ndcx * res.x - 0.5f;
            float prevy = ndcy * res.y - 0.5f;
            /////////////
            
            bool v[4];
            float floorx = floor(prevx);
            float floory = floor(prevy);
            float fracx = prevx - floorx;
            float fracy = prevy - floory;
            
            bool valid = (floorx >= 0 && floory >= 0 && floorx < res.x && floory < res.y);

            // 2x2 tap bilinear filter
            glm::ivec2 offset[4] = { glm::ivec2(0,0), glm::ivec2(1,0), glm::ivec2(0,1), glm::ivec2(1,1) };
            
            // check validity
            {
                for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++) {
                    glm::ivec2 loc = glm::ivec2(floorx, floory) + offset[sampleIdx];
                    v[sampleIdx] = isReprjValid(res, glm::ivec2(x, y), loc, current_gbuffer, prev_gbuffer);
                    valid = valid && v[sampleIdx];
                }
            }

            glm::vec3 prevColor = glm::vec3(0.0f);
            glm::vec2 prevMoments = glm::vec2(0.0f);
            float prevHistoryLength = 0.0f;

            if (valid) {
                // interpolate?
                float sumw = 0.0f;
                float w[4] = { (1 - fracx) * (1 - fracy),
                                fracx * (1 - fracy),
                                (1 - fracx) * fracy,
                                fracx * fracy };

                for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++) {
                    glm::ivec2 loc = glm::ivec2(floorx, floory) + offset[sampleIdx];
                    int locq = loc.x + loc.y * res.x;
                    if (v[sampleIdx]) {
                        prevColor += w[sampleIdx] * color_history[locq];
                        prevMoments += w[sampleIdx] * moment_history[locq];
                        prevHistoryLength += w[sampleIdx] * (float)history_length[locq];
                        sumw += w[sampleIdx];
                    }
                }
                if (sumw >= 0.01) {
                    prevColor /= sumw;
                    prevMoments /= sumw;
                    prevHistoryLength /= sumw;
                    //prevHistoryLength = 1;
                    valid = true;
                }
            }

            // find suitable samples elsewhere
            if (!valid) {
                float cnt = 0.0f;
                const int radius = 1;

                for (int yy = -radius; yy <= radius; yy++) {
                    for (int xx = -radius; xx <= radius; xx++) {
                        glm::vec2 loc = glm::vec2(floorx, floory) + glm::vec2(xx, yy);
                        int q = loc.x + res.x * loc.y;
                        if (isReprjValid(res, glm::ivec2(x, y), loc, current_gbuffer, prev_gbuffer)) {
                            prevColor += color_history[q];
                            prevMoments += moment_history[q];
                            prevHistoryLength += history_length[q];
                            cnt += 1.0f;
                        }
                    }
                }

                if (cnt > 0.0f) {
                    prevColor /= cnt;
                    prevMoments /= cnt;
                    prevHistoryLength /= cnt;
                    //prevHistoryLength = 0;
                    valid = true;
                }
            }

            if (valid) {
                // calculate alpha values that controls fade
                float color_alpha = max(1.0f / (float)(N + 1), color_alpha_min);
                float moment_alpha = max(1.0f / (float)(N + 1), moment_alpha_min);

                // incresase history length
                history_length_update[p] = (int)prevHistoryLength + 1;

                // color accumulation
                color_acc[p] = current_color[p] * color_alpha + prevColor * (1.0f - color_alpha);

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
        history_length_update[p] = 1;
        color_acc[p] = current_color[p];
        moment_acc[p] = glm::vec2(luminance, luminance * luminance);
        variacne_out[p] = 100.0f;
    }
}

// Estimate variance spatially
__global__ void EstimateVariance(float * variacne, glm::vec3 * color, glm::vec2 res) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < res.x && y < res.y) {
        int p = x + y * res.x;
        // TODO
        variacne[p] = 10.0f;
    }
}

template <typename T>
__global__ void DebugView(glm::ivec2 res, glm::vec3 * colorout, T * value, float scale) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < res.x && y < res.y) {
        int p = x + y * res.x;
        colorout[p] = glm::vec3((float)value[p] / scale);
    }
}

glm::mat4 GetViewMatrix(const Camera& cam) {
    return glm::inverse(glm::mat4(glm::vec4(cam.right,    0.f),
                                  glm::vec4(cam.up,       0.f),
                                  glm::vec4(cam.view,     0.f),
                                  glm::vec4(cam.position, 1.f)));
}

void denoise(glm::vec3 * output, glm::vec3 * input, GBufferTexel * gbuffer) {
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D blocks
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    /* Estimate Variance */
    float color_alpha = ui_temporal_enable ? ui_color_alpha : 1.0f;
    float moment_alpha = ui_temporal_enable ? ui_moment_alpha : 1.0f;
    if (ui_temporal_enable){
        BackProjection<<<blocksPerGrid2d, blockSize2d>>>(dev_variance, dev_history_length, dev_history_length_update, dev_moment_history, dev_color_history,
                                                                       dev_moment_acc, dev_color_acc, input, gbuffer, dev_gbuffer_prev, view_matrix_prev, cam.resolution,
                                                                       color_alpha, moment_alpha);
        cudaMemcpy(dev_color_history, dev_color_acc, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    }
    else {
        EstimateVariance<<<blocksPerGrid2d, blockSize2d>>>(dev_variance, dev_color_acc, cam.resolution);
        cudaMemcpy(dev_color_history, input, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    }

    if (ui_right_view_option == 1) {
        DebugView<<<blocksPerGrid2d, blockSize2d>>>(cam.resolution, output, dev_history_length, 100.0f);
    } 
    else if (ui_right_view_option == 2) {
        DebugView<<<blocksPerGrid2d, blockSize2d >>>(cam.resolution, output, dev_variance, 0.1f);
    }
    else {
        if (ui_atrous_nlevel == 0 || !ui_spatial_enable) {
            /* Skip A-Tours filter */
            cudaMemcpy(output, dev_color_history, sizeof(glm::vec3) * pixelcount, cudaMemcpyDeviceToDevice);
        }
        else {
            /* Apply A-Tours filter */            
            for (int level = 1; level <= ui_atrous_nlevel; level++) {
                glm::vec3* src = (level == 1) ? dev_color_history : dev_temp[level % 2];
                glm::vec3* dst = (level == ui_atrous_nlevel) ? output : dev_temp[(level + 1) % 2];
                ATrousFilter<<<blocksPerGrid2d, blockSize2d>>>(src, dst, dev_variance, gbuffer, cam.resolution, level, (level == ui_atrous_nlevel),
                                                               ui_sigmal, ui_sigman, ui_sigmax, ui_blurvariance, (ui_sepcolor && ui_addcolor));
                if (level == ui_history_level) cudaMemcpy(dev_color_history, dst, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
            }
        }
    }

    cudaMemcpy(dev_gbuffer_prev, gbuffer, sizeof(GBufferTexel) * pixelcount, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_moment_history, dev_moment_acc, sizeof(glm::vec2) * pixelcount, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_history_length, dev_history_length_update, sizeof(int) * pixelcount, cudaMemcpyDeviceToDevice);
    view_matrix_prev = GetViewMatrix(cam);

    cudaDeviceSynchronize();
}
