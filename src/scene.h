#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_obj_loader.h"
#include "boundingbox.h"
#include "bvhtree.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    //int loadMesh(string filename);
    int loadCamera();
	void loadLight();
	void loadMesh(string objPath, Geom& newGeom, const glm::mat4& transform, const glm::mat4& invTranspose);

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
	string file_name;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
	  
#if USE_KDTREE
	BVH_ArrNode *bvh_nodes;
	int Node_count = -1;
#endif
	int current_Triangle_id;  	
	std::vector<Triangle> triangles;
	std::vector<BoundingBox> BoudningBoxs;
	std::vector<Texture> textures;

	std::vector<Light> lights;
};
