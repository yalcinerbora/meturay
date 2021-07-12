#include "GuideDebugGL.h"
#include "GLFWCallbackDelegator.h"

#include "RayLib/VisorError.h"
#include "RayLib/Log.h"
#include "RayLib/FileSystemUtility.h"

// Debug Renderers
#include "GDebugRendererPPG.h"
//#include "GDebugRendererReference.h"

#include <filesystem>

void GuideDebugGL::OGLCallbackRender(GLenum,
                                     GLenum type,
                                     GLuint id,
                                     GLenum severity,
                                     GLsizei,
                                     const char* message,
                                     const void*)
{
    GLFWCallbackDelegator::OGLDebugLog(type, id, severity, message);
}

void GuideDebugGL::SetFBSizeFromInput(const Vector2i& fbs)
{
    viewportSize = fbs;
}

void GuideDebugGL::SetWindowSizeFromInput(const Vector2i& ws)
{
    windowSize = ws;
}

void GuideDebugGL::SetOpenStateFromInput(bool b)
{
    open = b;
}

VisorInputI* GuideDebugGL::InputInterface()
{
    return input;
}


GuideDebugGL::GuideDebugGL(const Vector2i& ws,
                           const std::u8string& guideDebugFile)
    : windowSize(ws)
    , viewportSize(ws)
    , glfwWindow(nullptr)
    , dummyVOpts{}
    , configFile(guideDebugFile)
    , configPath(std::filesystem::path(guideDebugFile).parent_path().string())
    , gradientTexture(Zero2ui, PixelFormat::RGB8_UNORM)
{

    // Initially Create Generator Map
    gdbGenerators.emplace(GDebugRendererPPG::TypeName,
                          GDBRendererGen(GDBRendererConstruct<GDebugRendererI, 
                                                              GDebugRendererPPG>));
    //gdbGenerators.emplace(GDebugRendererReference::TypeName,
    //                      GDBRendererGen(GDBRendererConstruct<GDebugRendererI, 
    //                                                          GDebugRendererReference>));





    bool configParsed = GuideDebug::ParseConfigFile(config, configFile);
    if(!configParsed) throw VisorException(VisorError::WINDOW_GENERATION_ERROR);

    VisorError e = Initialize();
    if(e != VisorError::OK) throw VisorException(e);
}

GuideDebugGL::~GuideDebugGL()
{
    if(glfwWindow != nullptr) glfwDestroyWindow(glfwWindow);
    GLFWCallbackDelegator::Instance().DetachWindow(glfwWindow);
}

VisorError GuideDebugGL::Initialize()
{
    GLFWCallbackDelegator& glfwCallback = GLFWCallbackDelegator::Instance();

    // Common Window Hints
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

    // This was buggy on nvidia cards couple of years ago
    // So instead manually convert image using
    // computer shader or w/e sRGB space
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_FALSE);

    glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_FLUSH);

    glfwWindowHint(GLFW_STEREO, GLFW_FALSE);

    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);

    // No depth buffer or stencil buffer etc
    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);

    glfwWindow = glfwCreateWindow(windowSize[0],
                                  windowSize[1],
                                  "METUray GuideDebug",
                                  nullptr,
                                  nullptr);

    if(glfwWindow == nullptr)
    {
        return VisorError::WINDOW_GENERATION_ERROR;
    }

    // Set Callbacks
    glfwCallback.AttachWindow(glfwWindow, this);

    glfwMakeContextCurrent(glfwWindow);
    glfwSwapInterval(0);

    // Now Init GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if(err != GLEW_OK)
    {
        METU_ERROR_LOG("%s", glewGetErrorString(err));
        return VisorError::RENDER_FUCTION_GENERATOR_ERROR;
    }

    // Print Stuff Now
    // Window Done
    METU_LOG("Window Initialized.");
    METU_LOG("GLEW\t: %s", glewGetString(GLEW_VERSION));
    METU_LOG("GLFW\t: %s", glfwGetVersionString());
    METU_LOG("");
    METU_LOG("Renderer Information...");
    METU_LOG("OpenGL\t: %s", glGetString(GL_VERSION));
    METU_LOG("GLSL\t: %s", glGetString(GL_SHADING_LANGUAGE_VERSION));
    METU_LOG("Device\t: %s", glGetString(GL_RENDERER));
    METU_LOG("");

    if constexpr(IS_DEBUG_MODE)
    {
        // Add Callback
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(GuideDebugGL::OGLCallbackRender, nullptr);
        glDebugMessageControl(GL_DONT_CARE,
                              GL_DONT_CARE,
                              GL_DONT_CARE,
                              0,
                              nullptr,
                              GL_TRUE);
    }

    // Pre-Bind Everything
    // States
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);


    // Generate Gradient Texture
    Vector2ui gradientDimensions = Vector2ui(1, config.gradientValues.size());
    std::vector<Byte> packedData(sizeof(Vector3f) * gradientDimensions[1]);
    memcpy(packedData.data(), config.gradientValues.data(), packedData.size());
    gradientTexture = TextureGL(gradientDimensions, PixelFormat::RGB8_UNORM);
    gradientTexture.CopyToImage(packedData,
                                Zero2ui, gradientDimensions, 
                                PixelFormat::RGB_FLOAT);

    // Generate DebugRenderers
    for(const auto& gc : config.guiderConfigs)
    {
        const std::string& guiderType = gc.first;
        const nlohmann::json& guiderConfig = gc.second;

        DebugRendererPtr gdbPtr = nullptr;
        auto loc = gdbGenerators.find(guiderType);
        if(loc == gdbGenerators.end()) return VisorError::NO_LOGIC_FOR_GUIDE_DEBUGGER;

        gdbPtr = loc->second(guiderConfig, gradientTexture);
        debugRenderers.emplace_back(std::move(gdbPtr));
    }


    // Create GUI
    gui = std::make_unique<GuideDebugGUI>(glfwWindow, 
                                          Utility::MergeFileFolder(configPath, config.refImage),
                                          debugRenderers);

    glfwShowWindow(glfwWindow);
    open = true;

    glfwMakeContextCurrent(nullptr);
    return VisorError::OK;
}

bool GuideDebugGL::IsOpen()
{
    return open;
}

void GuideDebugGL::Render()
{
    glfwMakeContextCurrent(glfwWindow);

    glClear(GL_COLOR_BUFFER_BIT);

    gui->Render();

    glfwSwapBuffers(glfwWindow);
}

// Input System
void GuideDebugGL::SetInputScheme(VisorInputI& i)
{
    input = &i;
}

// Misc
void GuideDebugGL::SetWindowSize(const Vector2i& size)
{
    glfwSetWindowSize(glfwWindow, size[0], size[1]);
}

void GuideDebugGL::SetFPSLimit(float)
{

}

Vector2i GuideDebugGL::MonitorResolution() const
{
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);

    return Vector2i(mode->width, mode->height);
}

void GuideDebugGL::SetRenderingContextCurrent()
{
    glfwMakeContextCurrent(glfwWindow);
}

void GuideDebugGL::ReleaseRenderingContext()
{
    glfwMakeContextCurrent(nullptr);
}


void GuideDebugGL::ProcessInputs()
{
    glfwPollEvents();
}