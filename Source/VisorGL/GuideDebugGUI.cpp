#include "ImageIO/EntryPoint.h"

#include "GuideDebugGUI.h"

#include <Imgui/imgui.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_opengl3.h>

#include "RayLib/Log.h"
#include "RayLib/HybridFunctions.h"
#include "RayLib/VisorError.h"

#include "GDebugRendererReference.h"

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

bool GuideDebugGUI::IncrementDepth()
{
    bool doInc = currentDepth < (MaxDepth - 1);
    if(doInc) currentDepth++;
    return doInc;
}

bool GuideDebugGUI::DecrementDepth()
{
    bool doDec = currentDepth > 0;
    if(doDec) currentDepth--;
    return doDec;
}

GuideDebugGUI::GuideDebugGUI(GLFWwindow* w,
                             uint32_t maxDepth,
                             const std::string& refFileName,
                             const std::string& posFileName,
                             const std::string& sceneName,
                             const std::vector<DebugRendererPtr>& dRenderers,
                             const GDebugRendererRef& dRef)
    : fullscreenShow(true)
    , window(w)
    , refTexture(refFileName)
    , debugRenderers(dRenderers)
    , pixelSelected(false)
    , sceneName(sceneName)
    , MaxDepth(maxDepth)
    , currentDepth(0)
    , debugReference(dRef)
    , debugRefTexture()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // ImGUI Dark
    ImGui::StyleColorsDark();
    
    float x, y;
    glfwGetWindowContentScale(window, &x, &y);

    // Initi renderer & platform
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(IMGUI_GLSL_STRING);

    // Scale everything according to the DPI
    assert(x == y);
    ImGui::GetStyle().ScaleAllSizes(x);
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    ImFontConfig config;
    config.SizePixels = std::roundf(14.0f * x);
    config.OversampleH = config.OversampleV = 2;
    config.PixelSnapH = true;    
    io.Fonts->AddFontDefault(&config);

    // Load Position Buffer
    Vector2ui size;
    PixelFormat pf;  
    std::vector<Byte> wpByte;
    ImageIOError e = ImageIOInstance().ReadImage(wpByte,
                                                 pf, size,
                                                 posFileName);
    
    if(e != ImageIOError::OK) throw ImageIOException(e);
    else if(pf != PixelFormat::RGB_FLOAT)
    {
        throw VisorException(VisorError::IMAGE_IO_ERROR,
                             "Reference Image Must have RGB format");
    }
    else if(size != refTexture.Size())
    {
        throw VisorException(VisorError::IMAGE_IO_ERROR,
                             "\"Position Image - Reference Image\" size mismatch");
    }
    // All fine, copy it to other vector
    else
    {
        worldPositions.resize(size[0] * size[1]);
        std::memcpy(reinterpret_cast<Byte*>(worldPositions.data()),
                    wpByte.data(),wpByte.size());
    }

    // Generate Textures
    for(size_t i = 0; i < debugRenderers.size(); i++)
    {
        guideTextues.emplace_back(Vector2ui(PG_TEXTURE_SIZE), PixelFormat::RGB8_UNORM);
        guidePixValues.emplace_back(PG_TEXTURE_SIZE * PG_TEXTURE_SIZE);
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
   
    // TODO: Properly do this no platform/lib dependent enums etc.
    // Check Input operation if left or right arrows are pressed to change the depth
    bool updateDirectionalTextures = false;
    if(ImGui::IsKeyReleased(GLFW_KEY_LEFT) &&
       DecrementDepth())
    {        
        updateDirectionalTextures = true;
        METU_LOG("Depth {:d}", currentDepth);
    }
    if(ImGui::IsKeyReleased(GLFW_KEY_RIGHT) &&
       IncrementDepth())
    {
        updateDirectionalTextures = true;
        METU_LOG("Depth {:d}", currentDepth);
    }
    
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
    ImGui::Dummy(ImVec2(0.0f, std::max(0.0f, paddingY - ImGui::GetFontSize()) * 0.95f));
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
    ImGui::Image(refTexId, refImgSize, ImVec2(0, 1), ImVec2(1, 0));    
    if(ImGui::IsItemHovered())
    {
        constexpr float ZOOM_FACTOR = 4.0f;

        ImGuiIO& io = ImGui::GetIO();

        // Zoomed Tooltip
        ImGui::BeginTooltip();
        float region_sz = 16.0f;
        float screenPixel_x = io.MousePos.x - refImgPos.x;
        float screenPixel_y = io.MousePos.y - refImgPos.y;
        
        // Calculate Actual Pixel
        float imagePixel_x = screenPixel_x * static_cast<float>(refTexture.Width()) / refImgSize.x;
        float imagePixel_y = screenPixel_y * static_cast<float>(refTexture.Height()) / refImgSize.y;        
        imagePixel_x = std::floor(imagePixel_x);
        imagePixel_y = std::floor(imagePixel_y);
        // Invert the Y axis
        imagePixel_y = refTexture.Height() - imagePixel_y - 1;
       
        ImGui::Text("Pixel: (%.2f, %.2f)", imagePixel_x, imagePixel_y);

        uint32_t linearIndex = (static_cast<uint32_t>(imagePixel_y) * refTexture.Size()[0] +
                                static_cast<uint32_t>(imagePixel_x));
        Vector4f worldPos = worldPositions[linearIndex];
        if(worldPos[3] == 0.0f)
        {
            worldPos[0] = std::numeric_limits<float>::infinity();
            worldPos[1] = std::numeric_limits<float>::infinity();
            worldPos[2] = std::numeric_limits<float>::infinity();
        }
        ImGui::Text("Position: (%.4f, %.4f, %.4f)",
                    worldPos[0], worldPos[1], worldPos[2]);

        // Calculate Zoom UV
        float region_x = imagePixel_x - region_sz * 0.5f;
        region_x = HybridFuncs::Clamp(region_x, 0.0f,
                                      refTexture.Width() - region_sz);
        float region_y = imagePixel_y - region_sz * 0.5f;
        region_y = HybridFuncs::Clamp(region_y, 0.0f,
                                      refTexture.Height() - region_sz);

        ImVec2 uv0 = ImVec2((region_x) / refTexture.Width(), 
                            (region_y) / refTexture.Height());        
        ImVec2 uv1 = ImVec2((region_x + region_sz) / refTexture.Width(), 
                            (region_y + region_sz) / refTexture.Height());
        // Invert Y (.......)
        std::swap(uv0.y, uv1.y);
       
        // Center the image on the tooltip window
        ImVec2 ttImgSize(region_sz * ZOOM_FACTOR,
                         region_sz * ZOOM_FACTOR);        
        ImGui::Image(refTexId, ttImgSize, uv0, uv1);
        ImGui::EndTooltip();

        // Click Action will require us to send draw calls to all Debug Renderers
        // Aswell we need to draw a circle
        if(ImGui::IsMouseClicked(ImGuiMouseButton_::ImGuiMouseButton_Left))
        {            
            pixelSelected = true;
            Vector2f newSelectedPixel = Vector2f(imagePixel_x, imagePixel_y);
            updateDirectionalTextures = (selectedPixel != newSelectedPixel);
            selectedPixel = newSelectedPixel;

        }
    }
    // Draw a circle on the selected location
    if(pixelSelected)
    {
        // Calculate Screen Pixel
        float screenPixel_x = selectedPixel[0] * refImgSize.x / static_cast<float>(refTexture.Width());
        float screenPixel_y = selectedPixel[1];
        screenPixel_y = refTexture.Height() - screenPixel_y - 1;
        screenPixel_y = screenPixel_y * refImgSize.y / static_cast<float>(refTexture.Height());

        // Draw a Circle on the Clicked Location
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        drawList->AddCircle(ImVec2(refImgPos.x + screenPixel_x + 0.5f,
                                   refImgPos.y + screenPixel_y + 0.5f),
                            3.0f,
                            ImColor(0.0f, 1.0f, 0.0f), 0, 1.5f);
    }   
    // Debug Reference Image
    ImTextureID dRefTexId = (void*)(intptr_t)debugRefTexture.TexId();
    ImGui::SameLine(0.0f, paddingX.x);
    ImGui::Image(dRefTexId, pgImgSize, ImVec2(0, 1), ImVec2(1, 0));
    // New Line and Path Guider Images
    ImGui::Dummy(ImVec2(0.0f, std::max(0.0f, (paddingY - ImGui::GetFontSize()) * 0.95f)));

    // Render Reference Images
    ImGui::SetCursorPosX(paddingX.y);
    float prevTextOffset = 0.0f;
    // TODO: More Generic PG Rendering
    assert(guideTextues.size() <= MAX_PG_IMAGE);
    assert(debugRenderers.size() <= MAX_PG_IMAGE);
    // Dummy here to prevent first same line to effect the other dummy
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
    for(size_t i = 0; i < guideTextues.size(); i++)
    {
        const std::string& name = debugRenderers[i]->Name();
        float offset = CenteredTextLocation(name.c_str(), pgImgSize.x);

        ImGui::SameLine(0.0f, prevTextOffset + paddingX.y + offset);
        prevTextOffset = offset;
        ImGui::Text(name.c_str());        
    }
    // Force New Line then render Images
    ImGui::NewLine();

    ImGui::SetCursorPosX(paddingX.y);
    for(size_t i = 0; i < guideTextues.size(); i++)
    {       
        ImTextureID guideTexId = (void*)(intptr_t) guideTextues[i].TexId();
        ImVec2 pgImgPos = ImGui::GetCursorScreenPos();        
        ImGui::Image(guideTexId, pgImgSize, ImVec2(0, 1), ImVec2(1, 0));
        ImGui::SameLine(0.0f, paddingX.y);

        if(ImGui::IsItemHovered())
        {
            constexpr float ZOOM_FACTOR = 4.0f;
            ImGuiIO& io = ImGui::GetIO();

            // Zoomed Tooltip
            ImGui::BeginTooltip();
            float region_sz = 16.0f;
            float screenPixel_x = io.MousePos.x - pgImgPos.x;
            float screenPixel_y = io.MousePos.y - pgImgPos.y;

            //METU_LOG("pgImgPos ({:f}, {:f}) == ({:f}, {:f}", 
            //         pgImgPos.x, pgImgPos.y,
            //         io.MousePos.x, io.MousePos.y);

            // Calculate Actual Pixel
            float imagePixel_x = screenPixel_x * static_cast<float>(PG_TEXTURE_SIZE) / pgImgSize.x;
            float imagePixel_y = screenPixel_y * static_cast<float>(PG_TEXTURE_SIZE) / pgImgSize.y;
            imagePixel_x = std::floor(imagePixel_x);
            imagePixel_y = std::floor(imagePixel_y);
            // Invert the Y axis
            imagePixel_y = PG_TEXTURE_SIZE - imagePixel_y - 1;


            uint32_t linearIndex = (PG_TEXTURE_SIZE * static_cast<uint32_t>(imagePixel_y)
                                    + static_cast<uint32_t>(imagePixel_x));
            ImGui::Text("Pixel: ({:.2f}, {:.2f})", imagePixel_x, imagePixel_y);
            ImGui::Text("Value: {:f}", guidePixValues[i][linearIndex]);

            // Calculate Zoom UV
            float region_x = imagePixel_x - region_sz * 0.5f;
            region_x = HybridFuncs::Clamp(region_x, 0.0f,
                                          PG_TEXTURE_SIZE - region_sz);
            float region_y = imagePixel_y - region_sz * 0.5f;
            region_y = HybridFuncs::Clamp(region_y, 0.0f,
                                          PG_TEXTURE_SIZE - region_sz);

            ImVec2 uv0 = ImVec2((region_x) / PG_TEXTURE_SIZE,
                                (region_y) / PG_TEXTURE_SIZE);
            ImVec2 uv1 = ImVec2((region_x + region_sz) / PG_TEXTURE_SIZE,
                                (region_y + region_sz) / PG_TEXTURE_SIZE);
            // Invert Y (.......)
            std::swap(uv0.y, uv1.y);

            // Center the image on the tooltip window
            ImVec2 ttImgSize(region_sz* ZOOM_FACTOR,
                             region_sz* ZOOM_FACTOR);
            ImGui::Image(guideTexId, ttImgSize, uv0, uv1);
            ImGui::EndTooltip();
        }

    }
    
    //// DEBUG TEST
    //ImGui::Dummy(ImVec2(0.0f, 0.0f));
    //for(size_t i = 0; i < 4; i++)
    //{
    //    const std::string name = std::string("PPG ") + std::to_string(i);        
    //    float offset = CenteredTextLocation(name.c_str(), pgImgSize.x);
    //    ImGui::SameLine(0.0f, prevTextOffset + paddingX.y + offset);
    //    prevTextOffset = offset;
    //    ImGui::Text(name.c_str());
    //}

    //ImGui::SetCursorPosX(paddingX.y);
    //for(size_t i = 0; i < 4; i++)
    //{
    //    ImTextureID refTexId = (void*)(intptr_t)refTexture.TexId();
    //    ImGui::Image(refTexId, pgImgSize);
    //    ImGui::SameLine(0.0f, paddingX.y);
    //}

    // Finish Window
    ImGui::End();

    // Before Render the Frame Update the Directional Textures if requested
    if(updateDirectionalTextures && pixelSelected)
    {
        uint32_t linearIndex = (static_cast<uint32_t>(selectedPixel[1]) * refTexture.Size()[0] +
                                static_cast<uint32_t>(selectedPixel[0]));
        Vector4f worldPos = worldPositions[linearIndex];
        if(worldPos[3] == 0.0f)
        {
            worldPos[0] = std::numeric_limits<float>::infinity();
            worldPos[1] = std::numeric_limits<float>::infinity();
            worldPos[2] = std::numeric_limits<float>::infinity();
        }

        debugReference.RenderDirectional(refTexture,
                                         Vector2i(static_cast<int32_t>(selectedPixel[0]),
                                                  static_cast<int32_t>(selectedPixel[1])),
                                         Vector2i(static_cast<int32_t>(refTexture.Size()[0]),
                                                  static_cast<int32_t>(refTexture.Size()[1])));

        // Do Guide Debuggers
        for(size_t i = 0; i < guideTextues.size(); i++)
        {
            debugRenderers[i]->RenderDirectional(guideTextues[i],
                                                 guidePixValues[i],
                                                 worldPos, currentDepth);
        }
    }
       
    // Render the GUI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}