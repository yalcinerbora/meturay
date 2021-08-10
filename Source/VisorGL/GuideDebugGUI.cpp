
#include "GuideDebugGUI.h"
#include <Imgui/imgui.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_opengl3.h>

#include "RayLib/Log.h"
#include "RayLib/HybridFunctions.h"
#include "RayLib/ImageIO.h"

GuideDebugGUI::GuideDebugGUI(GLFWwindow* w,
                             const std::string& refFileName,
                             const std::string& posFileName,
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


    // Load Position Buffer
    Vector2ui size;
    if(!ImageIO::Instance().ReadHDR(depthValues, size, posFileName))
    {
        METU_ERROR_LOG("Unable to Read Position Image");
    }
    if(size != refTexture.Size())
    {
        METU_ERROR_LOG("\"Position Image - Reference Image\" size mismatch");
    }
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
   
    //ImGui::ShowDemoWindow();

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

    //ImGui::BeginChild("Reference Image", refImageSize, false,
    //                  ImGuiWindowFlags_NoDecoration |
    //                  ImGuiWindowFlags_NoMove);

    ImVec2 refImageSize(static_cast<float>(refTexture.Width()),
                        static_cast<float>(refTexture.Height()));
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 uv_min = ImVec2(0.0f, 0.0f);                 // Top-left
    ImVec2 uv_max = ImVec2(1.0f, 1.0f);                 // Lower-right
    ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);   // No tint
    ImVec4 border_col = ImVec4(1.0f, 1.0f, 1.0f, 0.5f); // 50% opaque white

    ImGui::Image((void*)(intptr_t)refTexture.TexId(),
                 refImageSize);
    if(ImGui::IsItemHovered())
    {
        ImGuiIO& io = ImGui::GetIO();

        // Zoomed Tooltip
        ImGui::BeginTooltip();
        float region_sz = 16.0f;
        float pixel_x = io.MousePos.x - pos.x;        
        float pixel_y = io.MousePos.y - pos.y;
        
        float zoom = 5.0f;

        // Clamp Region
        float region_x = pixel_x - region_sz * 0.5f;
        region_x = HybridFuncs::Clamp(pixel_x, 0.0f,
                                      refTexture.Width() - region_sz);
        float region_y = pixel_y - region_sz * 0.5f;
        region_y = HybridFuncs::Clamp(pixel_y, 0.0f,
                                      refTexture.Height() - region_sz);
       
        ImGui::Text("Pixel: (%.2f, %.2f)", pixel_x, pixel_y);

        // Calculate Zoom UV;
        ImVec2 uv0 = ImVec2((region_x) / refTexture.Width(), 
                            (region_y) / refTexture.Height());
        ImVec2 uv1 = ImVec2((region_x + region_sz) / refTexture.Width(), 
                            (region_y + region_sz) / refTexture.Height());
        ImGui::Image((void*)(intptr_t)refTexture.TexId(), 
                     ImVec2(region_sz * zoom, region_sz * zoom), 
                     uv0, uv1);
        ImGui::EndTooltip();

        if(ImGui::IsMouseClicked(ImGuiMouseButton_::ImGuiMouseButton_Left))
        {
            METU_LOG("MouseClicked!");
        }

    }

    ////uv_min, uv_max, tint_col, border_col);
    //ImGui::EndChild();

    ImGui::End();







    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}