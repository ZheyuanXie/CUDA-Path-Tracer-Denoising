set(SOURCE_FILES
    "stb.cpp"
    "image.cpp"
    "image.h"
    "interactions.h"
    "intersections.h"
    "glslUtility.hpp"
    "glslUtility.cpp"
    "pathtrace.cu"
    "pathtrace.h"
    "scene.cpp"
    "scene.h"
    "sceneStructs.h"
    "preview.h"
    "preview.cpp"
    "utilities.cpp"
    "utilities.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
