#pragma once

#include <GL/glew.h>
#include <glfw/glfw3.h>
#include <vector>

#include "TextureGL.h"
#include "GuideDebugTypeGen.h"

struct ImVec2;

class GuideDebugGUI
{
    public:
        static constexpr const char*    IMGUI_GLSL_STRING = "#version 430 core";
        // GUI Texts
        static constexpr const char*    REFERENCE_TEXT = "Reference";
        // Constants & Limits 
        // (TODO: Dont limit this in future)
        static constexpr uint32_t       MAX_PG_IMAGE = 4;
        static constexpr uint32_t       PG_TEXTURE_SIZE = 1024;

    private:
        // Main Window
        GLFWwindow*                             window;
        // GUI Related
        const std::string&                      sceneName;
        bool                                    fullscreenShow;
        // Main texture that shows the scene
        TextureGL                               refTexture;        
        // Textures that are rendered by different visors
        std::vector<TextureGL>                  guideTextues;
        const std::vector<DebugRendererPtr>&    debugRenderers;
        
        // Reference Image's Pixel Values
        std::vector<Vector4f>                   depthValues;

        // Selected Pixel Location Related
        bool                                    initialSelection;
        Vector2f                                selectedPixel;

        static float                            CenteredTextLocation(const char* text, float centeringWidth);
        static void                             CalculateImageSizes(float& paddingY,
                                                                    ImVec2& paddingX,
                                                                    ImVec2& refImgSize,
                                                                    ImVec2& pgImgSize,
                                                                    const ImVec2& viewportSize);

    protected:

    public:
        // Constructors & Destructor
                        GuideDebugGUI(GLFWwindow* window,
                                      const std::string& refFileName,
                                      const std::string& posFileName,
                                      const std::string& sceneName,
                                      const std::vector<DebugRendererPtr>& dRenderers);
                        ~GuideDebugGUI();

        void            Render();
};