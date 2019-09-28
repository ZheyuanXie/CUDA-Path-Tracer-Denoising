#pragma once

#include <vector>
#include "scene.h"

#define CACHE_FIRST_ITERATION
//#define SORT_BY_MATERIALS
#define ANTI_ALIASING
//#define DEPTH_OF_FIELD

// thin lens camera model
constexpr float lens_radius = 0.5f;
constexpr float focal_distance = 5.5f;

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
