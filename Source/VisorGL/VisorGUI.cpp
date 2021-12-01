#include "VisorGUI.h"

#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_opengl3.h>
#include <Imgui/imgui_internal.h>
#include <glfw/glfw3.h>

#include "RayLib/FileSystemUtility.h"

// Icon Font UTF Definitions
#include "IcoMoonFontTable.h"

VisorGUI::VisorGUI(GLFWwindow* window)
    : bottomBarOn(true)
    , topBarOn(true)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    constexpr float INITIAL_SCALE = 1.05f;

    // Get Scale Info
    float x, y;
    glfwGetWindowContentScale(window, &x, &y);
    assert(x == y);

    x *= INITIAL_SCALE;
    constexpr float PIXEL_SIZE = 13;
    float scaledPixelSize = std::roundf(PIXEL_SIZE * x);

    // Set Scaled Fonts
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    ImFontConfig config;
    config.SizePixels = scaledPixelSize;
    config.PixelSnapH = false;
    config.MergeMode = false;
    std::string monoTTFPath = Utility::MergeFileFolder("Fonts", "VeraMono.ttf");
    io.Fonts->AddFontFromFileTTF(monoTTFPath.c_str(),
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
    io.Fonts->AddFontFromFileTTF(ofiTTFPath.c_str(),
                                 scaledPixelSize,
                                 &config, icon_ranges);

    // ImGUI Dark
    ImGui::StyleColorsDark();
    // Scale everything according to the DPI
    auto& style = ImGui::GetStyle();
    style.ScaleAllSizes(x);
    style.Colors[ImGuiCol_Button] = ImVec4(0, 0, 0, 0.1f);

    // Initi renderer & platform
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(IMGUI_GLSL_STRING);
}

VisorGUI::~VisorGUI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void VisorGUI::Render(const AnalyticData& ad,
                      const SceneAnalyticData& sad,
                      const Vector2i& resolution)
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
        statusBar.Render(ad, sad, resolution);
    }


    bool showDemo = true;
    //bool open     = true;
    ImGui::ShowDemoWindow(&showDemo);


    tmWindow.Render();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void VisorGUI::ChangeScene(std::u8string s)
{
    if(delegatedCallbacks) delegatedCallbacks->ChangeScene(s);
}

void VisorGUI::ChangeTime(double t)
{
    if(delegatedCallbacks) delegatedCallbacks->ChangeTime(t);
}

void VisorGUI::IncreaseTime(double t)
{
    if(delegatedCallbacks) delegatedCallbacks->IncreaseTime(t);
}

void VisorGUI::DecreaseTime(double t)
{
    if(delegatedCallbacks) delegatedCallbacks->DecreaseTime(t);
}

void VisorGUI::ChangeCamera(VisorTransform t)
{
    if(delegatedCallbacks) delegatedCallbacks->ChangeCamera(t);
}

void VisorGUI::ChangeCamera(unsigned int i)
{
    if(delegatedCallbacks) delegatedCallbacks->ChangeCamera(i);
}

void VisorGUI::StartStopTrace(bool b)
{
    if(delegatedCallbacks) delegatedCallbacks->StartStopTrace(b);
}

void VisorGUI::PauseContTrace(bool b)
{
    if(delegatedCallbacks) delegatedCallbacks->PauseContTrace(b);
}

void VisorGUI::WindowMinimizeAction(bool minimized)
{
    if(delegatedCallbacks) delegatedCallbacks->WindowMinimizeAction(minimized);
}

void VisorGUI::WindowCloseAction()
{
    if(delegatedCallbacks) delegatedCallbacks->WindowCloseAction();
}
