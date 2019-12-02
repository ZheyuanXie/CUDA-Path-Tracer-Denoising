#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_obj_loader.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadMesh(string filename);
    int loadCamera();

    glm::vec3 camera_position_default_;
    glm::vec3 camera_lookat_default_;
    glm::vec3 camera_up_default_;
public:
    Scene(string filename);
    ~Scene() {};

    void resetCamera() { 
        Camera &cam = state.camera;
        cam.position = camera_position_default_;
        cam.lookAt = camera_lookat_default_;
        cam.view = glm::normalize(cam.lookAt - cam.position);
        cam.up = camera_up_default_;
        cam.right = glm::normalize(glm::cross(cam.view, cam.up));
    }

    std::vector<Triangle> triangles;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
