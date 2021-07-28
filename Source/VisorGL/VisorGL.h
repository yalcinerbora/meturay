#pragma once
/**

OGL Implementation of Visor View
Uses GLFW glfw has c style interface and required to be initalized
at start of the program. We will need single window thus making
the VisorGL singleton.

*/

#include <GL/glew.h>
#include <glfw/glfw3.h>
#include <memory>
#include <mutex>

#include "RayLib/MPMCQueue.h"
#include "RayLib/ThreadVariable.h"

#include "ShaderGL.h"
#include "VisorGUI.h"
#include "ToneMapGL.h"
#include "WindowGLI.h"

// Basic command list implementation
struct VisorGLCommand
{
    public:
        enum Type
        {
            SET_PORTION,
            RESET_IMAGE,
            REALLOC_IMAGES,
            SAVE_IMAGE,
            SAVE_IMAGE_HDR
        };

    public:
        Type                    type;

        // Data will be
        std::vector<Byte>       data;
        PixelFormat             format;
        Vector2i                start;
        Vector2i                end;
        size_t                  offset;

        // Commands should not be copied
                                VisorGLCommand() = default;
                                VisorGLCommand(VisorGLCommand&&) = default;
                                VisorGLCommand(const VisorGLCommand&) = delete;
        VisorGLCommand&         operator=(const VisorGLCommand&) = delete;
        VisorGLCommand&         operator=(VisorGLCommand&&) = default;
};

class VisorGL : public WindowGLI
{
    private:
        static constexpr float      PostProcessTriData[6] =
        {
            3.0f, -1.0f,
            -1.0f, 3.0f,
            -1.0f, -1.0f
        };

        // Shader Location Cnstants
        // T: Texture Object
        // I: Image Object
        // IN: Shader Inputs
        // OUT: Shader Outputs
        // U: Uniforms
        static constexpr GLenum     T_IN_COLOR = 0;
        static constexpr GLenum     T_IN_BUFFER = 1;
        static constexpr GLenum     T_IN_SAMPLE = 2;

        static constexpr GLenum     I_OUT_COLOR = 0;
        static constexpr GLenum     I_SAMPLE = 1;

        static constexpr GLenum     IN_POS = 0;

        static constexpr GLenum     U_RES = 0;
        static constexpr GLenum     U_START = 1;
        static constexpr GLenum     U_END = 2;

    private:
        VisorInputI*                input;
        GLFWwindow*                 window;
        bool                        open;

        VisorOptions                vOpts;
        Vector2i                    imageSize;
        PixelFormat                 imagePixFormat;

        // GL Classes
        ToneMapGL                   toneMapGL;

        // Image portion list
        MPMCQueue<VisorGLCommand>   commandList;
        ThreadVariable<Vector2i>    viewportSize;

        // Texture Related
        GLuint                      outputTextures[2];
        GLuint                      sampleCountTexture;
        GLuint                      bufferTexture;
        GLuint                      sampleTexture;
        GLuint                      linearSampler;
        GLuint                      nearestSampler;
        GLuint                      sdrTexture;
        int                         currentIndex;

        // Shader
        ShaderGL                    vertPP;
        ShaderGL                    fragPP;
        ShaderGL                    compAccum;

        // Vertex
        GLuint                      vao;
        GLuint                      vBuffer;

        // GUI
        std::unique_ptr<VisorGUI>   visorGUI;

        // OGL Debug Context Callback
        static void                 OGLCallbackRender(GLenum source,
                                                      GLenum type,
                                                      GLuint id,
                                                      GLenum severity,
                                                      GLsizei length,
                                                      const char* message,
                                                      const void* userParam);

        // Image Allocation
        void                        ReallocImages();

        // Internal Command Handling
        void                        ProcessCommand(const VisorGLCommand&);
        void                        RenderImage();

         void                       SetFBSizeFromInput(const Vector2i&) override;
         void                       SetWindowSizeFromInput(const Vector2i&) override;
         void                       SetOpenStateFromInput(bool) override;
         VisorInputI*               InputInterface() override;

    protected:
    public:
        // Constructors & Destructor
                                VisorGL(const VisorOptions&,                                        
                                        const Vector2i& imgRes,
                                        const PixelFormat& imagePixelFormat);
                                VisorGL(const VisorGL&) = delete;
        VisorGL&                operator=(const VisorGL&) = delete;
                                ~VisorGL();

        // Interface
        VisorError              Initialize(VisorInputI&) override;

        bool                    IsOpen() override;
        void                    Render() override;
        // Data Related
        void                    SetImageRes(Vector2i resolution) override;
        void                    SetImageFormat(PixelFormat f) override;
        // Reset Data (Clears the RGB(A) Buffer of the Image)
        // and resets total accumulated rays
        void                    ResetSamples(Vector2i start = Zero2i,
                                             Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        // Append incoming data from
        void                    AccumulatePortion(const std::vector<Byte> data,
                                                  PixelFormat, size_t offset,
                                                  Vector2i start = Zero2i,
                                                  Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        // Options
        const VisorOptions&     VisorOpts() const override;
        // Misc
        void                    SetWindowSize(const Vector2i& size) override;
        void                    SetFPSLimit(float) override;
        Vector2i                MonitorResolution() const override;
        void                    SetCamera(const VisorCamera&) override;
        void                    SetSceneCameraCount(uint32_t) override;

        // Setting rendering context on current thread
        void                    SetRenderingContextCurrent() override;
        void                    ReleaseRenderingContext() override;
        // Main Thread only Calls
        void                    ProcessInputs() override;
        //
        void                    SaveImage(bool saveAsHDR) override;
};