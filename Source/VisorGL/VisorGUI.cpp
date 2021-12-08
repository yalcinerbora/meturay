#include "VisorGUI.h"

#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_opengl3.h>
#include <Imgui/imgui_internal.h>
#include <glfw/glfw3.h>

#include "RayLib/FileSystemUtility.h"

// Icon Font UTF Definitions
#include "IcoMoonFontTable.h"

VisorGUI::VisorGUI(VisorCallbacksI& cb,
                   bool& isWindowOpen,
                   Vector2i& windowSize,
                   Vector2i& viewportSize,
                   ToneMapOptions& tmOpts,
                   ImageSaverI& saver,
                   const Vector2i& imageSize,
                   const GLFWwindow* window,
                   const KeyboardKeyBindings& kb,
                   const MouseKeyBindings& mb,
                   MovementSchemeList&& mvList)
    : VisorWindowInput(cb, isWindowOpen,
                       windowSize, viewportSize, saver,
                       kb, mb, std::move(mvList))
    , tmWindow(tmOpts)
    , topBarOn(true)
    , bottomBarOn(true)
    , tracerAnalyticData{}
    , sceneAnalyticData{}
    , currentTOpts{}
    , currentTParams{}
    , imageSize(imageSize)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    constexpr float INITIAL_SCALE = 1.05f;

    // Get Scale Info
    float x, y;
    glfwGetWindowContentScale(const_cast<GLFWwindow*>(window), &x, &y);
    assert(x == y);

    x *= INITIAL_SCALE;
    constexpr float PIXEL_SIZE = 13;
    float scaledPixelSize = std::roundf(PIXEL_SIZE * x);

    // Set Scaled Fonts
    const std::string execPath = Utility::CurrentExecPath();

    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    ImFontConfig config;
    config.SizePixels = scaledPixelSize;
    config.PixelSnapH = false;
    config.MergeMode = false;
    std::string monoTTFPath = Utility::MergeFileFolder("Fonts", "VeraMono.ttf");
    std::string fullMonoTTFPath = Utility::MergeFileFolder(execPath, monoTTFPath);
    io.Fonts->AddFontFromFileTTF(fullMonoTTFPath.c_str(),
                                 config.SizePixels,
                                 &config);
    // Icomoon
    config.MergeMode = true;
    config.PixelSnapH = true;
    config.GlyphMinAdvanceX = scaledPixelSize;
    config.GlyphMaxAdvanceX = scaledPixelSize;
    config.OversampleH = config.OversampleV = 1;
    config.GlyphOffset = ImVec2(0, 4);
    static const ImWchar icon_ranges[] = {ICON_MIN_ICOMN, ICON_MAX_ICOMN, 0};
    std::string ofiTTFPath = Utility::MergeFileFolder("Fonts", FONT_ICON_FILE_NAME_ICOMN);
    std::string fullOfiTTFPath = Utility::MergeFileFolder(execPath, ofiTTFPath);
    io.Fonts->AddFontFromFileTTF(fullOfiTTFPath.c_str(),
                                 scaledPixelSize,
                                 &config, icon_ranges);

    // ImGUI Dark
    ImGui::StyleColorsDark();
    // Scale everything according to the DPI
    auto& style = ImGui::GetStyle();
    style.ScaleAllSizes(x);
    style.Colors[ImGuiCol_Button] = ImVec4(0, 0, 0, 0.1f);

    // Init renderer & platform
    ImGui_ImplGlfw_InitForOpenGL(const_cast<GLFWwindow*>(window), true);
    ImGui_ImplOpenGL3_Init(IMGUI_GLSL_STRING);
}

VisorGUI::~VisorGUI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void VisorGUI::RenderGUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if(ImGui::IsKeyPressed(GLFW_KEY_M))
        topBarOn = !topBarOn;
    if(ImGui::IsKeyPressed(GLFW_KEY_N))
        bottomBarOn = !bottomBarOn;

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar |
                                    ImGuiWindowFlags_NoSavedSettings |
                                    ImGuiWindowFlags_MenuBar;
    float height = ImGui::GetFrameHeight();
    if(topBarOn)
    {
        if(ImGui::BeginViewportSideBar("##MenuBar", NULL, ImGuiDir_Up, height, window_flags))
        {
            if(ImGui::BeginMenuBar())
            {
                if(ImGui::Button("Tone Mapping"))
                {
                    tmWindow.ToggleWindowOpen();
                }
                ImGui::EndMenuBar();
            }
        }
        ImGui::End();
    }

    if(bottomBarOn)
    {
        statusBar.Render(visorCallbacks,
                         tracerRunState,
                         tracerAnalyticData,
                         sceneAnalyticData,
                         imageSize);
    }


    //bool showDemo = true;
    //ImGui::ShowDemoWindow(&showDemo);

    tmWindow.Render();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void VisorGUI::SetSceneAnalyticData(const SceneAnalyticData& sad)
{
    sceneAnalyticData = sad;
}

void VisorGUI::SetTracerAnalyticData(const TracerAnalyticData& tad)
{
    tracerAnalyticData = tad;
}

void VisorGUI::SetTracerOptions(const TracerOptions& tOpts)
{
    currentTOpts = tOpts;
}

void VisorGUI::SetTracerParams(const TracerParameters& tParams)
{
    currentTParams = tParams;
}
