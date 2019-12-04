#include "main.h"
#include "preview.h"
#include <cstring>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene *scene;
RenderState *renderState;
int iteration;
int frame = 0;

int width;
int height;

float camera_tx;
float camera_ty;
float camera_tz;

// GUI state
bool ui_run = true;
bool ui_reset_denoiser = false;
float ui_sigmal = 0.001f;
float ui_sigmax = 0.2f;
float ui_sigman = 0.2f;
int ui_atrous_nlevel = 1;   // How man levels of A-trous filter used in denoising?
int ui_history_level = 0;   // Which level of A-trous output is sent to history buffer?
bool ui_accumulate = true;
bool ui_automate_camera = false;
float ui_camera_speed_x = 0.1;
float ui_camera_speed_y = 0.0;
float ui_camera_speed_z = 0.0;
bool ui_step = false;
float ui_color_alpha = 0.2;
float ui_moment_alpha = 0.2;
int ui_left_view_option = 0;
int ui_right_view_option = 0;
float ui_varpow = 1.0f;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage() {
    float samples = iteration;
    // output image file
    image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
    // camera automatic motion
    if (ui_automate_camera) {
        Camera &cam = renderState->camera;
        camera_tx += ui_camera_speed_x;
        camera_ty += ui_camera_speed_y;
        camera_tz += ui_camera_speed_z;
        cam.lookAt.x = sinf(camera_tx);
        cam.lookAt.y = 5.0f + sinf(camera_ty);
        cam.lookAt.z = 1.5f * sinf(camera_tz);
        camchanged = true;
    }

    if (camchanged) {
        //iteration = 0;
        Camera &cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
      }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    if (frame == 0 || ui_reset_denoiser == true) {
        denoiseFree();
        denoiseInit(scene);
        ui_reset_denoiser = false;
    }

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene);
    }

    if (iteration < 1) {
        uchar4 *pbo_dptr = NULL;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);
        pathtrace(pbo_dptr, frame++, ++iteration);        // execute the kernel
        cudaGLUnmapBufferObject(pbo);                   // unmap buffer object
    }

    if (iteration == 1) {
        iteration = 0;
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_ESCAPE:
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
      case GLFW_KEY_S:
        saveImage();
        break;
      case GLFW_KEY_SPACE:
        camchanged = true;
        renderState = &scene->state;
        Camera &cam = renderState->camera;
        cam.lookAt = ogLookAt;
        break;
      }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
  if (middleMousePressed) {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  }
  else if (rightMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f; 
    forward = glm::normalize(forward);

    cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  else if (leftMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 up = cam.up;
    up.x = 0.0f;
    up.z = 0.0f;
    up = glm::normalize(up);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * up * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}
