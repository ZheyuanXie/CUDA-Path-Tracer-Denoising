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
public:
    Scene(string filename);
    ~Scene();

    std::vector<Triangle> triangles;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
