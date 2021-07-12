#pragma once

#include <GL/glew.h>
#include <glfw/glfw3.h>

#include "WindowGLI.h"
#include "GuideDebugGUI.h"
#include "GuideDebugStructs.h"

class GuideDebugGL : public WindowGLI
{    
    private:
        const VisorOptions                      dummyVOpts;
        const std::u8string                     configFile;

        std::map<std::string, GDBRendererGen>   gdbGenerators;

        std::string                             configPath;
        GuideDebugConfig                        config;

        VisorInputI*                            input;
        GLFWwindow*                             glfwWindow;

        bool                                    open;
        Vector2i                                viewportSize;
        Vector2i                                windowSize;

        // OGL Types
        TextureGL                               gradientTexture;

        // Debugger Related
        std::vector<DebugRendererPtr>           debugRenderers;
        std::unique_ptr<GuideDebugGUI>          gui;

        static void             OGLCallbackRender(GLenum source,
                                                  GLenum type,
                                                  GLuint id,
                                                  GLenum severity,
                                                  GLsizei length,
                                                  const char* message,
                                                  const void* userParam);

        // Protected Interface
        void                    SetFBSizeFromInput(const Vector2i&) override;
        void                    SetWindowSizeFromInput(const Vector2i&) override;
        void                    SetOpenStateFromInput(bool) override;
        VisorInputI*            InputInterface() override;

        // Hidden Interface
        const VisorOptions&     VisorOpts() const override;
        void                    SetImageFormat(PixelFormat f) override;
        void                    SetImageRes(Vector2i resolution) override;
        void                    ResetSamples(Vector2i start = Zero2i,
                                             Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void                    AccumulatePortion(const std::vector<Byte> data,
                                                  PixelFormat, size_t offset,
                                                  Vector2i start = Zero2i,
                                                  Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void                    SetCamera(const VisorCamera&) override;
        void                    SetSceneCameraCount(uint32_t) override;

    protected:
    public:
        // Constructors & Destructor
                                GuideDebugGL(const Vector2i& windowSize,
                                             const std::u8string&);
                                GuideDebugGL(const GuideDebugGL&) = delete;
        GuideDebugGL&           operator=(const GuideDebugGL&) = delete;
                                ~GuideDebugGL();


        // Interface
        VisorError              Initialize() override;

        bool                    IsOpen() override;
        void                    Render() override;
        // Input System
        void                    SetInputScheme(VisorInputI&) override;
        // Misc
        void                    SetWindowSize(const Vector2i& size) override;
        void                    SetFPSLimit(float) override;
        Vector2i                MonitorResolution() const override;

        // Setting rendering context on current thread
        void                    SetRenderingContextCurrent() override;
        void                    ReleaseRenderingContext() override;
        // Main Thread only Calls
        void                    ProcessInputs() override;
};

// Some functions that are not necessary for GuideDebug
inline const VisorOptions& GuideDebugGL::VisorOpts() const { return dummyVOpts; }
inline void GuideDebugGL::SetImageFormat(PixelFormat) {}
inline void GuideDebugGL::SetImageRes(Vector2i){};
inline void GuideDebugGL::ResetSamples(Vector2i, Vector2i) {};
inline void GuideDebugGL::AccumulatePortion(const std::vector<Byte>,
                                          PixelFormat, size_t, Vector2i, Vector2i) {}
inline void GuideDebugGL::SetCamera(const VisorCamera&) {}
inline void GuideDebugGL::SetSceneCameraCount(uint32_t) {}