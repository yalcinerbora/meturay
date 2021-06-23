#pragma once

#include <GL/glew.h>
#include <glfw/glfw3.h>
#include <vector>

class GuideDebugGUI
{
    public:
        static constexpr const char* IMGUI_GLSL_STRING = "#version 430 core";

    private:
        bool                    fullscreenShow;
        GLFWwindow*             window;

        GLuint                  mainTexture;
        std::vector<GLuint>     guideTextues;

        // Main Image Aspect Ratio
        float                   ratio;

    protected:

    public:
        // Constructors & Destructor
                        GuideDebugGUI(GLFWwindow* window);
                        ~GuideDebugGUI();

        void            Render();
};