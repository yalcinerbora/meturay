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

bool VisorGUI::InitThread(GLFWwindow* window)
{
    //if(!init)
    //{
    //    init = true;
    //    IMGUI_CHECKVERSION();
    //    ImGui::CreateContext();
    //    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    //    // ImGUI Dark
    //    ImGui::StyleColorsDark();

    //    // Initi renderer & platform
    //    ImGui_ImplGlfw_InitForOpenGL(window, true);
    //    ImGui_ImplOpenGL3_Init(IMGUI_GLSL_STRING);
    //}
    //return init;
    return false;
}

void VisorGUI::RenderStart()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool showDemo = true;
    ImGui::ShowDemoWindow(&showDemo);

    ImGui::Render();
    
}

void VisorGUI::RenderEnd()
{
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void VisorGUI::ProcessInputs()
{
}