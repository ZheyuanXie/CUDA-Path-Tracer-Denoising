cmake_minimum_required(VERSION 3.1)

project(cis565_path_tracer)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
find_package(CUDA 10 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
########################################

find_package(OpenGL REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
else(UNIX)
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

set(headers
    src/main.h
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/scene.h
    src/sceneStructs.h
    src/preview.h
    src/utilities.h
    src/denoise.h
    src/boundingbox.h
    src/bvhtree.h
    )

set(sources
    src/main.cpp
    src/stb.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/scene.cpp
    src/preview.cpp
    src/utilities.cpp
    src/denoise.cu
    src/boundingbox.cpp
    src/bvhtree.cpp
    )

set(imgui
    src/imgui/imconfig.h
    src/imgui/imgui.cpp
    src/imgui/imgui.h
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_internal.h
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_demo.cpp
    src/imgui/imgui_impl_glfw.cpp
    src/imgui/imgui_impl_glfw.h
    src/imgui/imgui_impl_opengl2.cpp
    src/imgui/imgui_impl_opengl2.h
    src/imgui/imgui_impl_opengl3.cpp
    src/imgui/imgui_impl_opengl3.h
    )

set(tinyobjloader
    src/tinyobjloader/tiny_obj_loader.cc
    src/tinyobjloader/tiny_obj_loader.h
    )

list(SORT headers)
list(SORT sources)
list(SORT imgui)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})
source_group(ImGui FILES ${imgui})
source_group(TinyObjLoader FILES ${tinyobjloader})

cuda_add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui} ${tinyobjloader})
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${LIBRARIES}
    )
