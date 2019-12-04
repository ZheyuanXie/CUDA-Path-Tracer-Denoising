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
        ImGuiIO& io = ImGui::GetIO();

        // Setup Dear ImGui style
        ImGui::StyleColorsClassic();

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
    }

    return true;
}

void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Path trace 1 frame
        {
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
        }

        // Dear imgui new frame
        {
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
        }

        // Dear imgui define
        {
            ImGui::Begin("Console");
            ImGui::SetWindowFontScale(2);
            ImGui::Checkbox("Run", &ui_run);
            ImGui::SameLine();
            if (ImGui::Button("Step")) {
                ui_step = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear")) {
                ui_reset_denoiser = true;
            }
            ImGui::Checkbox("Accumulate", &ui_accumulate);
            
            if (ImGui::CollapsingHeader("Camera"))
            {
                Camera & cam = scene->state.camera;
                ImGui::Checkbox("Automate", &ui_automate_camera);
                ImGui::SameLine();
                if (ImGui::Button("Reset")) {
                    scene->resetCamera();
                }
                ImGui::SliderFloat("Spd. X", &ui_camera_speed_x, 0.0f, 0.5f);
                ImGui::SliderFloat("Spd. Y", &ui_camera_speed_y, 0.0f, 0.5f);
                ImGui::SliderFloat("Spd. Z", &ui_camera_speed_z, 0.0f, 0.5f);
                ImGui::Text("Camera Up: (%.3f, %.3f, %.3f)", cam.up.x, cam.up.y, cam.up.z);
                ImGui::Text("Camera Right: (%.3f, %.3f, %.3f)", cam.right.x, cam.right.y, cam.right.z);
                ImGui::Text("Camera View: (%.3f, %.3f, %.3f)", cam.view.x, cam.view.y, cam.view.z);
            }

            if (ImGui::CollapsingHeader("SVGF")) {
                ImGui::SliderFloat("C. Alpha", &ui_color_alpha, 0.0f, 1.0f);
                ImGui::SliderFloat("M. Alpha", &ui_moment_alpha, 0.0f, 1.0f);
                ImGui::Separator();
                ImGui::SliderFloat("Sigma L.", &ui_sigmal, 0.0f, 128.0f);
                ImGui::SliderFloat("Sigma X.", &ui_sigmax, 0.0f, 1.0f);
                ImGui::SliderFloat("Sigma N.", &ui_sigman, 0.0f, 1.0f);
                ImGui::SliderFloat("Var. Power", &ui_varpow, 0.5f, 5.0f);
                ImGui::SliderInt("# Lv.", &ui_atrous_nlevel, 0, 7);
                ImGui::SliderInt("Hist. Lv.", &ui_history_level, 0, ui_atrous_nlevel);
            } 

            if (ImGui::CollapsingHeader("View")) {
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

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
