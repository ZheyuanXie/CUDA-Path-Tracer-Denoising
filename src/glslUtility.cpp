// GLSL Utility: A utility class for loading GLSL shaders
// Written by Varun Sampath, Patrick Cozzi, and Karl Li.
// Copyright (c) 2012 University of Pennsylvania

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include "glslUtility.hpp"

using std::ios;

namespace glslUtility {

// embedded passthrough shaders so that default passthrough shaders don't need to be loaded
static std::string passthroughVS =
    "	attribute vec4 Position; \n"
    "	attribute vec2 Texcoords; \n"
    "	varying vec2 v_Texcoords; \n"
    "	\n"
    "	void main(void){ \n"
    "		v_Texcoords = Texcoords; \n"
    "		gl_Position = Position; \n"
    "	}";
static std::string passthroughFS =
    "	varying vec2 v_Texcoords; \n"
    "	\n"
    "	uniform sampler2D u_image; \n"
    "	\n"
    "	void main(void){ \n"
    "		gl_FragColor = texture2D(u_image, v_Texcoords); \n"
    "	}";

typedef struct {
    GLuint vertex;
    GLuint fragment;
    GLint geometry;
} shaders_t;

char* loadFile(const char *fname, GLint &fSize) {
    // file read based on example in cplusplus.com tutorial
    std::ifstream file (fname, ios::in | ios::binary | ios::ate);
    if (file.is_open()) {
        unsigned int size = (unsigned int)file.tellg();
        fSize = size;
        char *memblock = new char [size];
        file.seekg (0, ios::beg);
        file.read (memblock, size);
        file.close();
        std::cout << "file " << fname << " loaded" << std::endl;
        return memblock;
    }

    std::cout << "Unable to open file " << fname << std::endl;
    exit(EXIT_FAILURE);
}

// printShaderInfoLog
// From OpenGL Shading Language 3rd Edition, p215-216
// Display (hopefully) useful error messages if shader fails to compile
void printShaderInfoLog(GLint shader) {
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 1) {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
        std::cout << "InfoLog:" << std::endl << infoLog << std::endl;
        delete [] infoLog;
    }
}

void printLinkInfoLog(GLint prog) {
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 1) {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetProgramInfoLog(prog, infoLogLen, &charsWritten, infoLog);
        std::cout << "InfoLog:" << std::endl << infoLog << std::endl;
        delete [] infoLog;
    }
}

void compileShader(const char* shaderName, const char * shaderSource, GLenum shaderType, GLint &shaders) {
    GLint s;
    s = glCreateShader(shaderType);

    GLint slen = (unsigned int)std::strlen(shaderSource);
    char * ss = new char [slen + 1];
    std::strcpy(ss, shaderSource);

    const char * css = ss;
    glShaderSource(s, 1, &css, &slen);

    GLint compiled;
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        std::cout << shaderName << " did not compile" << std::endl;
    }
    printShaderInfoLog(s);

    shaders = s;

    delete [] ss;
}

shaders_t loadDefaultShaders() {
    shaders_t out;

    compileShader("Passthrough Vertex", passthroughVS.c_str(), GL_VERTEX_SHADER, (GLint&)out.vertex);
    compileShader("Passthrough Fragment", passthroughFS.c_str(), GL_FRAGMENT_SHADER, (GLint&)out.fragment);

    return out;
}

shaders_t loadShaders(const char * vert_path, const char * frag_path, const char * geom_path = 0) {
    shaders_t out;

    // load shaders & get length of each
    GLint vlen, flen, glen;
    char *vertexSource, *fragmentSource, *geometrySource;
    const char *vv, *ff, *gg;

    vertexSource = loadFile(vert_path, vlen);
    vv = vertexSource;
    compileShader("Vertex", vv, GL_VERTEX_SHADER, (GLint&)out.vertex);

    fragmentSource = loadFile(frag_path, flen);
    ff = fragmentSource;
    compileShader("Fragment", ff, GL_FRAGMENT_SHADER, (GLint&)out.fragment);

    if (geom_path) {
        geometrySource = loadFile(geom_path, glen);
        gg = geometrySource;
        compileShader("Geometry", gg, GL_GEOMETRY_SHADER, (GLint&)out.geometry);
    }

    return out;
}

void attachAndLinkProgram( GLuint program, shaders_t shaders) {
    glAttachShader(program, shaders.vertex);
    glAttachShader(program, shaders.fragment);

    glLinkProgram(program);
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        std::cout << "Program did not link." << std::endl;
    }
    printLinkInfoLog(program);
}

GLuint createDefaultProgram(const char *attributeLocations[], GLuint numberOfLocations) {
    glslUtility::shaders_t shaders = glslUtility::loadDefaultShaders();

    GLuint program = glCreateProgram();

    for (GLuint i = 0; i < numberOfLocations; ++i) {
        glBindAttribLocation(program, i, attributeLocations[i]);
    }

    glslUtility::attachAndLinkProgram(program, shaders);

    return program;
}

GLuint createProgram(const char *vertexShaderPath, const char *fragmentShaderPath,
                     const char *attributeLocations[], GLuint numberOfLocations) {
    glslUtility::shaders_t shaders = glslUtility::loadShaders(vertexShaderPath, fragmentShaderPath);

    GLuint program = glCreateProgram();

    for (GLuint i = 0; i < numberOfLocations; ++i) {
        glBindAttribLocation(program, i, attributeLocations[i]);
    }

    glslUtility::attachAndLinkProgram(program, shaders);

    return program;
}
}
