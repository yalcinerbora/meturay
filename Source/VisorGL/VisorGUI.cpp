#include "VisorGUI.h"

#include <glfw/glfw3.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_opengl3.h>

VisorGUI::VisorGUI(GLFWwindow* window)
{
    init = true;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // ImGUI Dark
    ImGui::StyleColorsDark();

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

void VisorGUI::Render(GLuint sdrTex, const Vector2i& resolution)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool showDemo = true;
    bool open = true;
    //ImGui::ShowDemoWindow(&showDemo);

    //ImGui::Begin("TestWindow", &open);

    //ImVec2 ws = ImGui::GetWindowSize();
    //ImGui::Image((void*)(intptr_t)sdrTex,
    //             ImVec2(ws.x - 10,
    //                    ws.y),
    //             //ImVec2(static_cast<float>(640),
    //                    //static_cast<float>(360)),
    //             //ImVec2(static_cast<float>(resolution[0]),
    //             //       static_cast<float>(resolution[1])),
    //             ImVec2(0, 1), ImVec2(1, 0));
    ////ImGui::SetWindowSize()
    //ImGui::SetWindowSize(ImVec2());

    //ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}