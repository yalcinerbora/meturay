#pragma once

#include <GL/glew.h>
#include <glfw/glfw3.h>

#include "RayLib/VisorI.h"

#include "GuideDebugGUI.h"
#include "GuideDebugStructs.h"
#include "GDebugRendererReference.h"

class GuideDebugGL : public VisorI
{
    private:
        const VisorOptions                      dummyVOpts;
        const std::u8string                     configFile;

        // GDB Generators
        std::map<std::string, GDBRendererGen>   gdbGenerators;

        std::string                             configPath;
        GuideDebugConfig                        config;

        std::unique_ptr<VisorInputI>            input;
        GLFWwindow*                             glfwWindow;

        bool                                    open;
        Vector2i                                viewportSize;
        Vector2i                                windowSize;

        // OGL Types
        std::unique_ptr<TextureGL>              gradientTexture;

        // Debugger Related
        std::unique_ptr<GDebugRendererRef>      referenceDebugRenderer;
        std::vector<DebugRendererPtr>           debugRenderers;
        std::unique_ptr<GuideDebugGUI>          gui;

        static void             OGLCallbackRender(GLenum source,
                                                  GLenum type,
                                                  GLuint id,
                                                  GLenum severity,
                                                  GLsizei length,
                                                  const char* message,
                                                  const void* userParam);
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
        //
        void                    SaveImage(bool saveAsHDR) override;
        //
        void                    Update(const VisorTransform&) override;
        void                    Update(uint32_t) override;
        void                    Update(const SceneAnalyticData&) override;
        void                    Update(const TracerAnalyticData&) override;
        void                    Update(const TracerOptions&) override;
        void                    Update(const TracerParameters&) override;

    protected:
    public:
        // Constructors & Destructor
                                GuideDebugGL(const Vector2i& windowSize,
                                             const std::u8string&);
                                GuideDebugGL(const GuideDebugGL&) = delete;
        GuideDebugGL&           operator=(const GuideDebugGL&) = delete;
                                ~GuideDebugGL();


        // Interface
        VisorError              Initialize(VisorCallbacksI& callbacks,
                                           const KeyboardKeyBindings& keyBindings,
                                           const MouseKeyBindings& mouseBindings,
                                           MovementSchemeList&& moveSchemeList) override;

        bool                    IsOpen() override;
        void                    Render() override;
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
inline void GuideDebugGL::SetImageRes(Vector2i){}
inline void GuideDebugGL::ResetSamples(Vector2i, Vector2i) {}
inline void GuideDebugGL::AccumulatePortion(const std::vector<Byte>,
                                          PixelFormat, size_t, Vector2i, Vector2i) {}
inline void GuideDebugGL::SaveImage(bool) {}

inline void GuideDebugGL::Update(const VisorTransform&) {}
inline void GuideDebugGL::Update(uint32_t) {}
inline void GuideDebugGL::Update(const SceneAnalyticData&) {}
inline void GuideDebugGL::Update(const TracerAnalyticData&) {}
inline void GuideDebugGL::Update(const TracerOptions&) {}
inline void GuideDebugGL::Update(const TracerParameters&) {}