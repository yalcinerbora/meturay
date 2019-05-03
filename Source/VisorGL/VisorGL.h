#pragma once
/**

OGL Implementation of Visor View
Uses GLFW glfw has c style interface and required to be initalized
at start of the program. We will need single window thus making
the VisorGL singleton.


*/

#include <gl\glew.h>
#include <glfw/glfw3.h>
#include <memory>
#include <mutex>

#include "RayLib/VisorI.h"
#include "RayLib/VisorInputI.h"

#include "RayLib/MPMCQueue.h"
#include "RayLib/ThreadData.h"

#include "ShaderGL.h"

// Basic command list implementation
struct VisorGLCommand
{
    public:
        enum Type
        {
            SET_PORTION,
            RESET_IMAGE,
            REALLOC_IMAGES
        };

    public:
        Type                    type;

        // Data will be
        std::vector<Byte>       data;
        PixelFormat             format;
        Vector2i                start;
        Vector2i                end;
        int                     sampleCount;

        // Commands should not be copied
        VisorGLCommand() = default;
        VisorGLCommand(VisorGLCommand&&) = default;
        VisorGLCommand(const VisorGLCommand&) = delete;
        VisorGLCommand& operator=(const VisorGLCommand&) = delete;
        VisorGLCommand& operator=(VisorGLCommand&&) = default;
};

class VisorGL : public VisorI
{
    private:
        static VisorGL*             instance;

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

        static constexpr GLenum     I_OUT_COLOR = 0;
        static constexpr GLenum     I_SAMPLE = 1;

        static constexpr GLenum     IN_POS = 0;
        static constexpr GLenum     IN_UV = 0;

        static constexpr GLenum     U_RES = 0;
        static constexpr GLenum     U_START = 1;
        static constexpr GLenum     U_END = 2;
        static constexpr GLenum     U_SAMPLE = 3;

    private:
        //VisorCallbacksI*            callbacks;
        VisorInputI*                input;
        GLFWwindow*                 window;
        bool                        open;

        VisorOptions                vOpts;

        // Image portion list
        MPMCQueue<VisorGLCommand>   commandList;
        ThreadData<Vector2i>        viewportSize;

        // Texture Related
        GLuint                      outputTextures[2];
        GLuint                      sampleCountTexture;
        GLuint                      bufferTexture;
        GLuint                      linearSampler;
        int                         currentIndex;
 
        // Shader
        ShaderGL                    vertPP;
        ShaderGL                    fragPP;
        ShaderGL                    compAccum;

        // Vertex
        GLuint                      vao;
        GLuint                      vBuffer;


        static KeyAction            DetermineAction(int);
        static MouseButtonType      DetermineMouseButton(int);
        static KeyboardKeyType      DetermineKey(int);

        // Callbacks
        // GLFW
        static void                 ErrorCallbackGLFW(int, const char*);
        static void                 WindowPosGLFW(GLFWwindow*, int, int);
        static void                 WindowFBGLFW(GLFWwindow*, int, int);
        static void                 WindowSizeGLFW(GLFWwindow*, int, int);
        static void                 WindowCloseGLFW(GLFWwindow*);
        static void                 WindowRefreshGLFW(GLFWwindow*);
        static void                 WindowFocusedGLFW(GLFWwindow*, int);
        static void                 WindowMinimizedGLFW(GLFWwindow*, int);

        static void                 KeyboardUsedGLFW(GLFWwindow*, int, int, int, int);
        static void                 MouseMovedGLFW(GLFWwindow*, double, double);
        static void                 MousePressedGLFW(GLFWwindow*, int, int, int);
        static void                 MouseScrolledGLFW(GLFWwindow*, double, double);

        // OGL Debug Context Callback
        static void __stdcall       OGLCallbackRender(GLenum source,
                                                      GLenum type,
                                                      GLuint id,
                                                      GLenum severity,
                                                      GLsizei length,
                                                      const char* message,
                                                      const void* userParam);
        // Visor to OGL conversions
        static GLenum               PixelFormatToGL(PixelFormat);
        static GLenum               PixelFormatToSizedGL(PixelFormat);
        static GLenum               PixelFormatToTypeGL(PixelFormat);

        // Image Allocation
        void                        ReallocImages();

        // Internal Command Handling
        void                        ProcessCommand(const VisorGLCommand&);
        void                        RenderImage();

    protected:
    public:
        // Constructors & Destructor
                                VisorGL(const VisorOptions&);
                                VisorGL(const VisorGL&) = delete;
        VisorGL&                operator=(const VisorGL&) = delete;
                                ~VisorGL();

        // Interface
        bool                    IsOpen() override;
        void                    Render() override;
        void                    ProcessInputs() override;
        // Input System
        void                    SetInputScheme(VisorInputI&) override;
        // Data Related
        void                    SetImageRes(Vector2i resolution) override;
        void                    SetImageFormat(PixelFormat f) override;
        // Reset Data (Clears the RGB(A) Buffer of the Image)
        // and resets total accumulated rays
        void                    ResetSamples(Vector2i start = Zero2i,
                                             Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        // Append incoming data from
        void                    AccumulatePortion(const std::vector<Byte> data,
                                                  PixelFormat, int sampleCount,
                                                  Vector2i start = Zero2i,
                                                  Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        // Options
        const VisorOptions&     VisorOpts() const override;
        // Misc

        void                    SetWindowSize(const Vector2i& size) override;
        void                    SetFPSLimit(float) override;
};