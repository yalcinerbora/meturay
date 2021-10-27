#pragma once

#include <GL/glew.h>
#include <glfw/glfw3.h>
#include <vector>

#include "TextureGL.h"
#include "GuideDebugTypeGen.h"

struct ImVec2;
class GDebugRendererRef;

class GuideDebugGUI
{
    public:
        static constexpr const char*    IMGUI_GLSL_STRING = "#version 430 core";
        // GUI Texts
        static constexpr const char*    REFERENCE_TEXT = "Reference";
        static constexpr const char*    OPTIONS_TEXT = "Options";
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
        std::vector<std::vector<float>>         guidePixValues;
        std::vector<TextureGL>                  guideTextues;
        const std::vector<DebugRendererPtr>&    debugRenderers;
        // Reference Texture
        std::vector<float>              debugRefPixValues;
        TextureGL                       debugRefTexture;
        const GDebugRendererRef&        debugReference;
        // Reference Image's Pixel Values
        std::vector<Vector3f>           worldPositions;
        // Current Depth Value
        const uint32_t                  MaxDepth;
        uint32_t                        currentDepth;
        // Generic Options
        bool                            doLogScale;

        // Selected Pixel Location Related
        bool                            pixelSelected;
        Vector2f                        selectedPixel;

        bool            IncrementDepth();
        bool            DecrementDepth();

        static ImVec2   FindRemainingSize(const ImVec2& size);
        static float    CenteredTextLocation(const char* text, float centeringWidth);
        static void     CalculateImageSizes(float& paddingY,
                                            ImVec2& paddingX,
                                            ImVec2& optionsSize,
                                            ImVec2& refImgSize,
                                            ImVec2& refPGImgSize,
                                            ImVec2& pgImgSize,
                                            const ImVec2& viewportSize);

        template<class T>
        static std::enable_if_t<std::is_same_v<T, Vector3f> ||
                                std::is_same_v<T, float>, std::tuple<bool, Vector2f>>
                        RenderImageWithZoomTooltip(TextureGL&,
                                                   const std::vector<T>& values,
                                                   const ImVec2& size,
                                                   bool renderCircle = false,
                                                   const Vector2f& circleTexel = Zero2f);

    protected:

    public:
        // Constructors & Destructor
                        GuideDebugGUI(GLFWwindow* window,
                                      uint32_t maxDepth,
                                      const std::string& refFileName,
                                      const std::string& posFileName,
                                      const std::string& sceneName,
                                      const std::vector<DebugRendererPtr>& dRenderers,
                                      const GDebugRendererRef&);
                        ~GuideDebugGUI();

        void            Render();
};