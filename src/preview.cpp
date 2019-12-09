#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include "main.h"
#include "preview.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#define IMGUI_IMPL_OPENGL_LOADER_GLEW

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow *window;

std::string currentTimeString() {
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width * 2, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
    GLfloat vertices[] = {
        // left display
        -1.0f, -1.0f,
        0.0f, -1.0f,
        0.0f,  1.0f,
        -1.0f,  1.0f,
        // right display
        0.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        0.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        // left display
        0.5f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        0.5f, 0.0f,
        // right display
        1.0f, 1.0f,
        0.5f, 1.0f,
        0.5f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2, 4, 5, 7, 7, 5, 6 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
    const char *attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda() {
    if (pbo) {
        deletePBO(&pbo);
    }
    if (displayImage) {
        deleteTexture(&displayImage);
    }
}

void initCuda() {
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO() {
    // set up vertex data parameter
    int num_texels = width * 2 * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);

}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "%s\n", description);
}

bool init() {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(width * 2, height, "CUDA Path Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSwapInterval(1); // Enable vsync

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    {

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
    }

    return true;
}

static ImGuiWindowFlags windowFlags= ImGuiWindowFlags_None | ImGuiWindowFlags_NoMove;
static bool ui_autoresize = true;
static bool ui_hide = false;

void drawGui(int windowWidth, int windowHeight) {
    // Dear imgui new frame
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    // Dear imgui define
    {
        ImVec2 minSize(300.f, 600.f);
        ImVec2 maxSize((float)windowWidth * 0.5, (float)windowHeight);
        ImGui::SetNextWindowSizeConstraints(minSize, maxSize);

        ImGui::SetNextWindowPos(ui_hide ? ImVec2(-1000.f, -1000.f) : ImVec2(0.0f, 0.0f));

        if (ImGui::IsKeyPressed('H')) {
            ui_hide = !ui_hide;
        }

        ImGui::Begin("Control Panel", 0, windowFlags);
        ImGui::SetWindowFontScale(1);

        // Capture keyboard
        if (ImGui::IsKeyPressed(' ')) {
            ImGui::SetWindowCollapsed(!ImGui::IsWindowCollapsed());
        }
        
        ImGui::Checkbox("Auto-Resize", &ui_autoresize);
        if (ui_autoresize) {
            windowFlags |= ImGuiWindowFlags_AlwaysAutoResize;
        } else {
            windowFlags &= ~ImGuiWindowFlags_AlwaysAutoResize;
        }

        if (ImGui::CollapsingHeader("Ray Tracing", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Run", &ui_run);
            ImGui::SameLine();
            if (ImGui::Button("Step")) {
                ui_step = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear")) {
                ui_reset_denoiser = true;
            }
            ImGui::SliderInt("Max. Depth", &ui_tracedepth, 1, 10);
            ImGui::Checkbox("Use KD-Tree", &ui_usekdtree);
            ImGui::Separator();
            ImGui::Checkbox("Trace Shadow Ray", &ui_shadowray);
            ImGui::SameLine();
            ImGui::Checkbox("Reduce Var.", &ui_reducevar);
            ImGui::SliderFloat("SR Int.", &ui_sintensity, 0.0f, 20.0f);
            ImGui::SliderFloat("Sample Rad.", &ui_lightradius, 0.0f, 2.0f);
        }

        if (ImGui::CollapsingHeader("Denosing", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::IsKeyPressed('D')) {
                ui_denoise_enable = !ui_denoise_enable;
            }
            else if (ImGui::IsKeyPressed('T')) {
                ui_temporal_enable = !ui_temporal_enable;
            }
            else if (ImGui::IsKeyPressed('F')) {
                ui_spatial_enable = !ui_spatial_enable;
            }
            if (ImGui::Checkbox("Enable(D)", &ui_denoise_enable)) {
                if (ui_denoise_enable) {

                }
                else {
                    camchanged = true;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("None")) {
                ui_temporal_enable = false;
                ui_spatial_enable = false;
                ui_sepcolor = false;
                ui_addcolor = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("All")) {
                ui_temporal_enable = true;
                ui_spatial_enable = true;
                ui_sepcolor = true;
                ui_addcolor = true;
            }
            ImGui::Checkbox("Temporal(T)", &ui_temporal_enable);
            ImGui::SameLine();
            ImGui::Checkbox("Spatial(F)", &ui_spatial_enable);
            ImGui::Checkbox("Rmv 1st Albedo", &ui_sepcolor);
            ImGui::SameLine();
            ImGui::Checkbox("Add 1st Albedo", &ui_addcolor);
            ImGui::Separator();
            ImGui::Text("Temporal Acc.");
            ImGui::SliderFloat("C. Alpha", &ui_color_alpha, 0.0f, 1.0f);
            ImGui::SliderFloat("M. Alpha", &ui_moment_alpha, 0.0f, 1.0f);
            if (ImGui::Button("Set Default Param.##1")) {
                ui_color_alpha = 0.2f;
                ui_moment_alpha = 0.2f;
            }
            ImGui::Separator();
            ImGui::Text("Variance Est.");
            ImGui::Checkbox("Blur Var.", &ui_blurvariance);
            ImGui::SliderFloat("Sigma L.", &ui_sigmal, 0.0f, 2.0f);
            ImGui::SliderFloat("Sigma X.", &ui_sigmax, 0.0f, 1.0f);
            ImGui::SliderFloat("Sigma N.", &ui_sigman, 0.0f, 1.0f);
            if (ImGui::Button("Set Default Param.##2")) {
                ui_sigmal = 0.7f;
                ui_sigmax = 0.35f;
                ui_sigman = 0.2f;
            }
            ImGui::Separator();
            ImGui::Text("A-Trous Wavelet");
            ImGui::SliderInt("Num. Lv.", &ui_atrous_nlevel, 0, 7);
            ImGui::SliderInt("Hist. Lv.", &ui_history_level, 0, ui_atrous_nlevel);
        }

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::IsKeyPressed('Z') || ImGui::Button("Reset Camera(Z)")) {
                scene->resetCamera();
                resetCamera();
            }

            if (ImGui::TreeNodeEx("Camera Info", ImGuiTreeNodeFlags_DefaultOpen)) {
                Camera & cam = scene->state.camera;
                ImGui::Text("Camera Up: (%.3f, %.3f, %.3f)", cam.up.x, cam.up.y, cam.up.z);
                ImGui::Text("Camera Right: (%.3f, %.3f, %.3f)", cam.right.x, cam.right.y, cam.right.z);
                ImGui::Text("Camera View: (%.3f, %.3f, %.3f)", cam.view.x, cam.view.y, cam.view.z);
                ImGui::Text("Camera Pos: (%.3f, %.3f, %.3f)", cam.position.x, cam.position.y, cam.position.z);
                ImGui::Text("Theta: %.3f, Phi: %.3f", theta, phi);
                ImGui::Text("Zoom: %.3f", zoom);
                ImGui::TreePop();
            }
            ImGui::Separator();
            ImGui::SliderFloat("Spd. X", &ui_camera_speed_x, 0.0f, 1.5f);
            ImGui::SliderFloat("Spd. Y", &ui_camera_speed_y, 0.0f, 0.5f);
            ImGui::SliderFloat("Spd. Z", &ui_camera_speed_z, 0.0f, 0.5f);
            ImGui::SliderFloat("Spd. Theta", &ui_camera_speed_theta, 0.0f, 0.5f);
            ImGui::SliderFloat("Spd. Phi", &ui_camera_speed_phi, 0.0f, 0.5f);
            if (ImGui::IsKeyPressed('A')) {
                ui_automate_camera = !ui_automate_camera;
            }
            ImGui::Checkbox("Automate Camera Motion(A)", &ui_automate_camera);
        }

        if (ImGui::CollapsingHeader("Debug View", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* listbox_items_left[] = { "1 spp" };
            ImGui::ListBox("Left View", &ui_left_view_option, listbox_items_left, IM_ARRAYSIZE(listbox_items_left), 3);
            const char* listbox_items_right[] = { "Filtered", "HistoryLenth", "Variance" };
            ImGui::ListBox("Right View", &ui_right_view_option, listbox_items_right, IM_ARRAYSIZE(listbox_items_right), 3);
        }

        ImGui::End();
    }

    // Dear imgui render
    {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
}

void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Path trace 1 frame
        if (ui_run || ui_step) {
            runCuda();
            if (ui_step) {
                ui_step = false;
            }
        }
        string title = "CUDA Path Tracer | Frame " + utilityCore::convertIntToString(frame);
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width * 2, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 12,  GL_UNSIGNED_SHORT, 0);

        // Draw imgui
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        drawGui(display_w, display_h);

        // Display content
        glViewport(0, (display_h - display_w / 2) / 2, display_w, display_w / 2);
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
