#pragma once

#include <vector>
#include "scene.h"

#define CACHE_FIRST_ITERATION
//#define SORT_BY_MATERIALS
//#define STREAM_COMPACTION

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
