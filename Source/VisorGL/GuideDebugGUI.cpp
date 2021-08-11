
#include "GuideDebugGUI.h"
#include <Imgui/imgui.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_opengl3.h>

#include "RayLib/Log.h"
#include "RayLib/HybridFunctions.h"
#include "RayLib/ImageIO.h"

float GuideDebugGUI::CenteredTextLocation(const char* text, float centeringWidth)
{
    float widthText = ImGui::CalcTextSize(text).x;
    // Handle overflow
    if(widthText > centeringWidth) return 0;
    
    return (centeringWidth - widthText) * 0.5f;
}

void GuideDebugGUI::CalculateImageSizes(float& paddingY,
                                        ImVec2& paddingX,
                                        ImVec2& refImgSize,
                                        ImVec2& pgImgSize,
                                        const ImVec2& viewportSize)
{
    // TODO: Dont assume that we would have at most 4 pg images (excluding ref pg image)
    constexpr float REF_IMG_ASPECT = 16.0f / 9.0f;
    constexpr float PADDING_PERCENT_Y = 0.05f;
    
    // Calculate Y padding between refImage and pgImages
    // Calculate ySize of the both images
    paddingY = viewportSize.y * PADDING_PERCENT_Y;
    float ySize = (viewportSize.y - 3.0f * paddingY) * 0.5f;
    // PG Images are always square so pgImage has size of ySize, ySize
    pgImgSize = ImVec2(ySize, ySize);
    // Ref images are always assumed to have 16/9 aspect
    float xSize = ySize * REF_IMG_ASPECT;
    refImgSize = ImVec2(xSize, ySize);

    // X value of padding X is the upper padding between refImage and ref PG Image
    // Y value is the bottom x padding between pg images
    paddingX.x = (viewportSize.x - refImgSize.x - pgImgSize.x) * (1.0f / 3.0f);
    paddingX.y = (viewportSize.x - MAX_PG_IMAGE * pgImgSize.x) / (MAX_PG_IMAGE + 2);
}

GuideDebugGUI::GuideDebugGUI(GLFWwindow* w,
                             const std::string& refFileName,
                             const std::string& posFileName,
                             const std::string& sceneName,
                             const std::vector<DebugRendererPtr>& dRenderers)
    : fullscreenShow(true)
    , window(w)
    , refTexture(refFileName)
    , debugRenderers(dRenderers)
    , initialSelection(false)
    , sceneName(sceneName)
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
    if(!ImageIO::Instance().ReadEXR(depthValues, size, posFileName))
    {
        // TODO: create VisorException for this
        METU_ERROR_LOG("Unable to Read Position Image");
     
        std::abort();
    }
    if(size != refTexture.Size())
    {
        // TODO: create VisorException for this
        METU_ERROR_LOG("\"Position Image - Reference Image\" size mismatch");
        std::abort();
    }

    // Generate Textures
    for(size_t i = 0; i < debugRenderers.size(); i++)
        guideTextues.emplace_back(Vector2ui(PG_TEXTURE_SIZE), PixelFormat::RGB8_UNORM);
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

    float paddingY;
    ImVec2 paddingX;
    ImVec2 refImgSize;
    ImVec2 pgImgSize;
    ImVec2 vpSize = ImVec2(viewport->Size.x - viewport->Pos.x,
                           viewport->Size.y - viewport->Pos.y);
    CalculateImageSizes(paddingY, paddingX, refImgSize, pgImgSize, vpSize);

    ImGui::Begin("guideDebug", &fullscreenShow,
                 ImGuiWindowFlags_NoBackground |
                 ImGuiWindowFlags_NoDecoration | 
                 ImGuiWindowFlags_NoMove | 
                 ImGuiWindowFlags_NoResize | 
                 ImGuiWindowFlags_NoSavedSettings);

    // Titles
    // Scene Name
    ImGui::Dummy(ImVec2(0.0f, std::max(0.0f, paddingY - ImGui::GetFontSize())));
    float sceneNameOffset = CenteredTextLocation(sceneName.c_str(), refImgSize.x);
    ImGui::SetCursorPosX(paddingX.x + sceneNameOffset);
    ImGui::Text(sceneName.c_str());

    // Reference Path Guiding Image Name
    float refTextOffset = CenteredTextLocation(REFERENCE_TEXT, pgImgSize.x);
    ImGui::SameLine(0.0f, sceneNameOffset + paddingX.x + refTextOffset);
    ImGui::Text(REFERENCE_TEXT);

    // Reference Image Texture
    ImTextureID refTexId = (void*)(intptr_t)refTexture.TexId();
    ImGui::SetCursorPosX(paddingX.x);
    ImVec2 refImgPos = ImGui::GetCursorScreenPos();
    ImGui::Image(refTexId, refImgSize);
    if(ImGui::IsItemHovered())
    {
        constexpr float ZOOM_FACTOR = 4.0f;

        ImGuiIO& io = ImGui::GetIO();

        // Zoomed Tooltip
        ImGui::BeginTooltip();
        float region_sz = 16.0f;
        float screenPixel_x = io.MousePos.x - refImgPos.x;
        float screenPixel_y = io.MousePos.y - refImgPos.y;

        // Clamp Region
        float region_x = screenPixel_x - region_sz * 0.5f;
        region_x = HybridFuncs::Clamp(screenPixel_x, 0.0f,
                                      refTexture.Width() - region_sz);
        float region_y = screenPixel_y - region_sz * 0.5f;
        region_y = HybridFuncs::Clamp(screenPixel_y, 0.0f,
                                      refTexture.Height() - region_sz);
       

        // Calculate Actual Pixel
        float imagePixel_x = screenPixel_x * static_cast<float>(refTexture.Width()) / refImgSize.x;
        float imagePixel_y = screenPixel_y * static_cast<float>(refTexture.Height()) / refImgSize.y;

        ImGui::Text("Pixel: (%.2f, %.2f)", imagePixel_x, imagePixel_y);

        uint32_t linearIndex = (static_cast<uint32_t>(imagePixel_y) * refTexture.Size()[0] +
                                static_cast<uint32_t>(imagePixel_x));
        Vector4f worldPos = depthValues[linearIndex];
        if(worldPos[3] == 0.0f)
        {
            worldPos[0] = std::numeric_limits<float>::infinity();
            worldPos[1] = std::numeric_limits<float>::infinity();
            worldPos[2] = std::numeric_limits<float>::infinity();
        }
        ImGui::Text("Position: (%.4f, %.4f, %.4f)",
                    worldPos[0], worldPos[1], worldPos[2]);

        // Calculate Zoom UV;
        ImVec2 uv0 = ImVec2((region_x) / refTexture.Width(), 
                            (region_y) / refTexture.Height());
        ImVec2 uv1 = ImVec2((region_x + region_sz) / refTexture.Width(), 
                            (region_y + region_sz) / refTexture.Height());


        // Center the image on the tooltip window
        ImVec2 ttImgSize(region_sz * ZOOM_FACTOR,
                         region_sz * ZOOM_FACTOR);        
        ImGui::Image(refTexId, ttImgSize, uv0, uv1);
        ImGui::EndTooltip();

        // Click Action will require us to send draw calls to all Debug Renderers
        // Aswell we need to draw a circle
        if(ImGui::IsMouseClicked(ImGuiMouseButton_::ImGuiMouseButton_Left))
        {            
            initialSelection = true;
            selectedPixel = Vector2f(imagePixel_x, imagePixel_y);
        }

    }
    // Draw a circle on the selected location
    if(initialSelection)
    {
        // Calculate Screen Pixel
        float imagePixel_x = selectedPixel[0] * refImgSize.x / static_cast<float>(refTexture.Width());
        float imagePixel_y = selectedPixel[1] * refImgSize.y / static_cast<float>(refTexture.Height());

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        drawList->AddCircle(ImVec2(refImgPos.x + imagePixel_x,
                                   refImgPos.y + imagePixel_y),
                            3.0f,
                            ImColor(0.0f, 1.0f, 0.0f), 0, 1.5f);
    }   
    // Reference Image
    ImGui::SameLine(0.0f, paddingX.x);
    ImGui::Image(refTexId, pgImgSize);
    // New Line and Path Guider Images
    ImGui::Dummy(ImVec2(0.0f, std::max(0.0f, paddingY - ImGui::GetFontSize())));

    // Render Reference Images
    ImGui::SetCursorPosX(paddingX.y);
    float prevTextOffset = 0.0f;
    // TODO: More Generic PG Rendering
    assert(guideTextues.size() <= MAX_PG_IMAGE);
    assert(debugRenderers.size() <= MAX_PG_IMAGE);
    // Dummy here to prevent first same line to effect the other dummy
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
    //for(size_t i = 0; i < guideTextues.size(); i++)
    //{
    //    const std::string& name = debugRenderers[i]->Name();
    //    float offset = CenteredTextLocation(name.c_str(), pgImgSize.x);

    //    ImGui::SameLine(0.0f, prevTextOffset + paddingX.y + offset);
    //    prevTextOffset = offset;
    //    ImGui::Text(name.c_str());        
    //}
    //// Force New Line then render Images
    //ImGui::NewLine();

    //ImGui::SetCursorPosX(paddingX.y);
    //for(size_t i = 0; i < guideTextues.size(); i++)
    //{       
    //    ImTextureID refTexId = (void*)(intptr_t) guideTextues[i].TexId();
    //    ImGui::Image(refTexId, pgImgSize);
    //    ImGui::SameLine(0.0f, paddingX.y);
    //}
    
    // DEBUG TEST
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
    for(size_t i = 0; i < 4; i++)
    {
        const std::string name = std::string("PPG ") + std::to_string(i);        
        float offset = CenteredTextLocation(name.c_str(), pgImgSize.x);
        ImGui::SameLine(0.0f, prevTextOffset + paddingX.y + offset);
        prevTextOffset = offset;
        ImGui::Text(name.c_str());
    }

    ImGui::SetCursorPosX(paddingX.y);
    for(size_t i = 0; i < 4; i++)
    {
        ImTextureID refTexId = (void*)(intptr_t)refTexture.TexId();
        ImGui::Image(refTexId, pgImgSize);
        ImGui::SameLine(0.0f, paddingX.y);
    }


    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}