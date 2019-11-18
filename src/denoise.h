#pragma once

#include "scene.h"
#include "pathtrace.h"

void denoiseInit(Scene *scene);
void denoiseFree();
void denoise(int iter, glm::vec3 * input, glm::vec3 * output, GBufferTexel * gbuffer);
