#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "glslUtility.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>

#include "sceneStructs.h"
#include "image.h"
#include "pathtrace.h"
#include "utilities.h"
#include "scene.h"
#include "denoise.h"

using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

extern Scene* scene;
extern int frame;

extern int width;
extern int height;

extern float zoom, theta, phi;
extern bool camchanged;

extern bool ui_run;
extern bool ui_reset_denoiser;
extern float ui_sigmal;
extern float ui_sigmax;
extern float ui_sigman;
extern int ui_atrous_nlevel;
extern int ui_history_level;
extern bool ui_automate_camera;
extern float ui_camera_speed_x;
extern float ui_camera_speed_y;
extern float ui_camera_speed_z;
extern bool ui_step;
extern int ui_step_target;
extern float ui_color_alpha;
extern float ui_moment_alpha;
extern int ui_left_view_option;
extern int ui_right_view_option;
extern int ui_tracedepth;
extern bool ui_denoise_enable;
extern bool ui_temporal_enable;
extern bool ui_spatial_enable;
extern bool ui_shadowray;
extern float ui_sintensity;
extern float ui_lightradius;
extern bool ui_blurvariance;
extern bool ui_reducevar;
extern float ui_camera_speed_theta;
extern float ui_camera_speed_phi;

void runCuda();
void resetCamera();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
