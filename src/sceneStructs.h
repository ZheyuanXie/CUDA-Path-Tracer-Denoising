#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtx/intersect.hpp"
#include "utilities.h"
#include <stb_image.h>
#include "boundingbox.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define ENABLE_MIS_LIGHTING 0;
#define USE_KDTREE 1
#define SHOW_TEXTURE 1

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
};

struct Vertex {
    Point3f pos;
    Normal3f normal;
    UV2f uv;
    Vertex() {};
    Vertex(Point3f p, Normal3f n) : pos(p), normal(n) { uv = UV2f(0.0); };
    Vertex(Point3f p, Normal3f n, UV2f _uv) : pos(p), normal(n), uv(_uv) {};
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    // Mesh
    int T_startidx;             // RECORD THE START AND END INDEX FOR VARIOUS OF MESH
    int T_endidx;
    int BoundIdx;               // BOUNDING BOX IN THE WORLD SPACE
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    int matid;                  // ADDED BY YAN
    int texid;                  // LOAD TEXTURE PART!
    int norid;

    Material() {
        color = glm::vec3(0.f);
        specular.color = glm::vec3(0.f);
        matid = -1;
        texid = -1;             // if id is -1 then not have any texture!
        emittance = 1;
        indexOfRefraction = 1.0f;
    }
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    bool diffuse;
    bool specular;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  int geomId;
  bool outside;
  glm::vec2 uv;
};

struct GBufferTexel {
    glm::vec3 normal;
    glm::vec3 position;
    glm::vec3 albedo;
    glm::vec3 ialbedo;
    int geomId;
};

struct Triangle {
    int id;
    Vertex verts[3];		//	THE THREE VERTEXES OF THE TRIANGLE
    Normal3f normal;		//	THE NORMAL OF THE TRIANGLE ITS SELF	

    struct {
        Point3f maxCorner;	// TRIANGLE MAX XYZ
        Point3f minCorner;	// TRIANGLE	MIN XYZ
    } boundingbox;

    Triangle() {};

    __host__ __device__ Point3f calculateMidPoint() {
        return (boundingbox.maxCorner + boundingbox.minCorner) / 2.f;
    }

    __host__ __device__ void calculateMinandMax(glm::vec3 &min, glm::vec3 &max) const {
        min.x = glm::min(verts[0].pos.x, glm::min(verts[1].pos.x, verts[2].pos.x));
        min.y = glm::min(verts[0].pos.y, glm::min(verts[1].pos.y, verts[2].pos.y));
        min.z = glm::min(verts[0].pos.z, glm::min(verts[1].pos.z, verts[2].pos.z));

        max.x = glm::max(verts[0].pos.x, glm::max(verts[1].pos.x, verts[2].pos.x));
        max.y = glm::max(verts[0].pos.y, glm::max(verts[1].pos.y, verts[2].pos.y));
        max.z = glm::max(verts[0].pos.z, glm::max(verts[1].pos.z, verts[2].pos.z));
    }

    __host__ __device__ BoundingBox GetWorldBoundbox() {
        glm::vec3 min(-1.f), max(-1.f);
        calculateMinandMax(min, max);
        return BoundingBox(min, max);
    }

    __host__ __device__ float Area() {
        return glm::length(glm::cross(verts[0].pos - verts[1].pos, verts[2].pos - verts[1].pos)) * 0.5f;
    }

    __host__ __device__ bool Intersect(const Ray& r, ShadeableIntersection* isect) const {
        glm::vec3 bari(0.f);
        if (glm::intersectRayTriangle(r.origin, r.direction, verts[0].pos, verts[1].pos, verts[2].pos, bari)) {
            isect->t = bari.z;
            // bari: alpha + beta + gamma = 1
            isect->uv = verts[0].uv * (1.0f - bari.x - bari.y) +
                verts[1].uv * bari.x +
                verts[2].uv * bari.y;

            //isect->uv = verts[0].uv * bari.x + verts[1].uv * bari.y + verts[2].uv * (1.f - bari.x - bari.y);
            //isect->surfaceNormal = verts[0].normal * (1.0f - bari.x - bari.y) + verts[1].normal * bari.x + verts[2].normal * bari.y;
            isect->surfaceNormal = verts[0].normal * bari.x +
                verts[1].normal * bari.y +
                verts[2].normal * (1.f - bari.x - bari.y);

            isect->surfaceNormal = glm::normalize(isect->surfaceNormal);
            glm::vec3 objspaceIntersection = r.origin + bari.z * glm::normalize(r.direction);
            return true;
        }
        else {
            isect->t = -1.0f;
            return false;
        }
    }
};

//TEXTURE
struct Texture {
    int width;
    int height;
    int components;
    unsigned char *image;
    unsigned char *dev_image;

    Texture() {
        width = -1;
        height = -1;
        components = -1;
        image = NULL;
    }

    void Load(char const *filename) {
        image = stbi_load(filename, &width, &height, &components, 0);
        if (image == NULL) {
            printf("FAILED TO LOAD TEXTURE!\n");
        }
        else {
            printf("SUCCESS TO LOAD TEXTURE~\n");
        }
    }

    __host__ __device__ glm::vec3 getColor(glm::vec2& uv) {
        int X = glm::min(1.f * width * uv.x, 1.f * width - 1.0f);
        int Y = glm::min(1.f * height * (1.0f - uv.y), 1.f * height - 1.0f);
        int texel_index = Y * width + X;
        if (components == 3) {
            glm::vec3 col = glm::vec3(dev_image[texel_index * components],
                dev_image[texel_index * components + 1],
                dev_image[texel_index * components + 2]);
            col = COLORDIVIDOR * col;
            //printf("texture color: %f, %f, %f", col.x, col.y, col.z);
            return col;
        }
        //return glm::vec3(-1.0f);
    }
};

// LIGHT PART
enum LightType {
    AREALIGHT,
    SPOT,
    POINT,
};

struct Light {
    enum LightType type;
    Geom geom;
    int geomIdx;
    int matIdx;
};
