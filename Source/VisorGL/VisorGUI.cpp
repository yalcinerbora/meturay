#include "VisorGUI.h"

#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_opengl3.h>
#include <Imgui/imgui_internal.h>
#include <glfw/glfw3.h>

#include "RayLib/FileSystemUtility.h"
#include "OpenFontIconsTable.h"

VisorGUI::VisorGUI(GLFWwindow* window)
    : bottomBarOn(false)
    , topBarOn(false)
{
    init = true;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Get Scale Info
    float x, y;
    glfwGetWindowContentScale(window, &x, &y);
    assert(x == y);

    // Set Scaled Fonts
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    ImFontConfig config;
    config.SizePixels = std::roundf(14.0f * x);
    config.PixelSnapH = false;
    config.MergeMode = false;
    std::string monoTTFPath = Utility::MergeFileFolder("Fonts", "VeraMono.ttf");
    io.Fonts->AddFontFromFileTTF(monoTTFPath.c_str(),
                                 config.SizePixels,
                                 &config);
    // Font Awesome
    config.MergeMode = true;
    config.PixelSnapH = false;
    config.GlyphMinAdvanceX = config.SizePixels;
    config.GlyphMaxAdvanceX = config.SizePixels;
    static const ImWchar icon_ranges[] = {ICON_MIN_OFI, ICON_MAX_OFI, 0};
    std::string ofiTTFPath = Utility::MergeFileFolder("Fonts", FONT_ICON_FILE_NAME_OFI);
    io.Fonts->AddFontFromFileTTF(ofiTTFPath.c_str(),
                                 config.SizePixels * 0.90f,
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

void VisorGUI::Render(GLuint, const Vector2i&)
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
                ImGui::Text("Happy top menu bar");
                ImGui::EndMenuBar();
            }
        }
        ImGui::End();
    }

    if(bottomBarOn)
    {
        if(ImGui::BeginViewportSideBar("##MainStatusBar", NULL, ImGuiDir_Down, height, window_flags))
        {
            if(ImGui::BeginMenuBar())
            {
                ImGui::Button(ICON_OFI_PLAY_CIRCLE);
                ImGui::Button(ICON_OFI_PAUSE_CIRCLE);
                ImGui::Button(ICON_OFI_STOP_CIRCLE);

                ImGui::Text("Happy status bar");
                ImGui::EndMenuBar();
            }
        }
        ImGui::End();
    }


    bool showDemo = true;
    //bool open     = true;
    ImGui::ShowDemoWindow(&showDemo);

    // ImGui::Begin("TestWindow", &open);

    // ImVec2 ws = ImGui::GetWindowSize();
    // ImGui::Image((void*)(intptr_t)sdrTex,
    //             ImVec2(ws.x - 10,
    //                    ws.y),
    //             //ImVec2(static_cast<float>(640),
    //                    //static_cast<float>(360)),
    //             //ImVec2(static_cast<float>(resolution[0]),
    //             //       static_cast<float>(resolution[1])),
    //             ImVec2(0, 1), ImVec2(1, 0));
    ////ImGui::SetWindowSize()
    // ImGui::SetWindowSize(ImVec2());

    // ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}