#pragma once

#include <GL/glew.h>
#include <glfw/glfw3.h>
#include <vector>

#include "TextureGL.h"

class GuideDebugGUI
{
    public:
        static constexpr const char* IMGUI_GLSL_STRING = "#version 430 core";

    private:
        // Main Window
        GLFWwindow*             window;

        // GUI Related
        bool                    fullscreenShow;

        // Main texture that shows the scene
        TextureGL               refTexture;
        // Textures that are rendered by different visors
        std::vector<TextureGL>  guideTextues;

        // Main Image Aspect Ratio
        float                   ratio;

    protected:

    public:
        // Constructors & Destructor
                        GuideDebugGUI(GLFWwindow* window,
                                      const std::string& refFileName);
                        ~GuideDebugGUI();

        void            Render();
};