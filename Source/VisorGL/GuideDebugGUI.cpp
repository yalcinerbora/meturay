
#include "GuideDebugGUI.h"
#include <Imgui/imgui.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_opengl3.h>


GuideDebugGUI::GuideDebugGUI(GLFWwindow* w,
                             const std::string& refFileName,
                             const std::vector<DebugRendererPtr>& dRenderers)
    : fullscreenShow(true)
    , window(w)
    , refTexture(refFileName)
    , ratio(static_cast<float>(refTexture.Width()) / static_cast<float>(refTexture.Height()))
    , debugRenderers(dRenderers)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // ImGUI Dark
    ImGui::StyleColorsDark();

    // Initi renderer & platform
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(IMGUI_GLSL_STRING);

}

GuideDebugGUI::~GuideDebugGUI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void GuideDebugGUI::Render()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
   
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);

    ImGui::Begin("guideDebug", &fullscreenShow,
                 ImGuiWindowFlags_NoBackground |
                 ImGuiWindowFlags_NoDecoration | 
                 ImGuiWindowFlags_NoMove | 
                 ImGuiWindowFlags_NoResize | 
                 ImGuiWindowFlags_NoSavedSettings);

    ImGui::Text("TESTOOO");

    
    ImGui::End();
    //ImGui::Image((void*)(intptr_t) mainTexture,
    //             ImVec2(my_tex_w, my_tex_h), uv_min, uv_max, tint_col, border_col);










    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}