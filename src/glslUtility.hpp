// GLSL Utility: A utility class for loading GLSL shaders
// Written by Varun Sampath, Patrick Cozzi, and Karl Li.
// Copyright (c) 2012 University of Pennsylvania

#ifndef GLSLUTILITY_HPP
#define GLSLUTILITY_HPP

#include <GL/glew.h>

namespace glslUtility {

GLuint createDefaultProgram(const char *attributeLocations[], GLuint numberOfLocations);
GLuint createProgram(const char *attributeLocations[], GLuint numberOfLocations,
                     const char *vertexShaderPath, const char *fragmentShaderPath,
                     const char *geometryShaderPath = 0);
}

#endif