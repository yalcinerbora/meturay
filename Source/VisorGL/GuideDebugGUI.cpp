#include "ImageIO/EntryPoint.h"

#include "GuideDebugGUI.h"

#include <Imgui/imgui.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_opengl3.h>

#include <Imgui/imgui_tex_inspect.h>
#include <Imgui/tex_inspect_opengl.h>

#include "RayLib/Log.h"
#include "RayLib/VisorError.h"

#include "GDebugRendererReference.h"
#include "GuideDebugGUIFuncs.h"


void GuideDebugGUI::CalculateImageSizes(float& paddingY,
                                        ImVec2& paddingX,
                                        ImVec2& optionsSize,
                                        ImVec2& refImgSize,
                                        ImVec2& refPGImgSize,
                                        ImVec2& pgImgSize,
                                        const ImVec2& viewportSize)
{
    // TODO: Dont assume that we would have at most 4 pg images (excluding ref pg image)
    constexpr float REF_IMG_ASPECT = 16.0f / 9.0f;
    constexpr float PADDING_PERCENT = 0.01f;

    // Calculate Y padding between refImage and pgImages
    // Calculate ySize of the both images
    paddingY = viewportSize.y * PADDING_PERCENT;
    float ySize = (viewportSize.y - 3.0f * paddingY) * 0.5f;

    // PG Images are always square so pgImage has size of ySize, ySize
    refPGImgSize = ImVec2(ySize, ySize);
    // Ref images are always assumed to have 16/9 aspect
    float xSize = ySize * REF_IMG_ASPECT;
    refImgSize = ImVec2(xSize, ySize);

    optionsSize = ImVec2(ySize * (3.0f / 4.5f), ySize);

    paddingX.y = viewportSize.x * PADDING_PERCENT;
    pgImgSize.x = (viewportSize.x - (MAX_PG_IMAGE + 1) * paddingX.y) / static_cast<float>(MAX_PG_IMAGE);
    pgImgSize.y = pgImgSize.x;

    // X value of padding X is the upper padding between refImage and ref PG Image
    // Y value is the bottom x padding between pg images
    paddingX.x = (viewportSize.x - refImgSize.x - refPGImgSize.x - optionsSize.x);
    paddingX.x *= (1.0f / 4.0f);
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
                             GDebugRendererRef& dRef,
                             const std::vector<DebugRendererPtr>& dRenderers)
    : fullscreenShow(true)
    , window(w)
    , refTexture(refFileName)
    , debugRenderers(dRenderers)
    , pixelSelected(false)
    , sceneName(sceneName)
    , MaxDepth(maxDepth)
    , currentDepth(0)
    , debugReference(dRef)
    , doLogScale(false)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // ImGUI Dark
    ImGui::StyleColorsDark();
    ImGuiTexInspect::ImplOpenGL3_Init();
    ImGuiTexInspect::Init();
    ImGuiTexInspect::CreateContext();

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
                             "Position Image Must have RGB format");
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
                    wpByte.data(), wpByte.size());
    }
}

GuideDebugGUI::~GuideDebugGUI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    //ImGuiTexInspect::DestroyContext();
    ImGui::DestroyContext();
    ImGuiTexInspect::Shutdown();
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
    }
    if(ImGui::IsKeyReleased(GLFW_KEY_RIGHT) &&
       IncrementDepth())
    {
        updateDirectionalTextures = true;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);

    float paddingY;
    ImVec2 paddingX;
    ImVec2 refImgSize;
    ImVec2 refPGImgSize;
    ImVec2 pgImgSize;
    ImVec2 optionsSize;
    ImVec2 vpSize = ImVec2(viewport->Size.x - viewport->Pos.x,
                           viewport->Size.y - viewport->Pos.y);
    CalculateImageSizes(paddingY, paddingX, optionsSize, refImgSize, refPGImgSize, pgImgSize, vpSize);

    // Start Main Window
    ImVec2 remainingSize;

    ImGui::Begin("guideDebug", &fullscreenShow,
                 ImGuiWindowFlags_NoBackground |
                 ImGuiWindowFlags_NoDecoration |
                 ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoSavedSettings);

    // Y Padding
    ImGui::Dummy(ImVec2(0.0f, std::max(0.0f, (paddingY - ImGui::GetFontSize()) * 0.95f)));
    ImGui::NewLine();

    // Options pane
    ImGui::SameLine(0.0f, paddingX.x);
    ImGui::BeginChild("optionsPane", optionsSize, true);
    ImGui::Text("Depth: %u", currentDepth);
    ImGui::NewLine();
    ImGui::SameLine(0.0f, GuideDebugGUIFuncs::CenteredTextLocation(OPTIONS_TEXT, optionsSize.x));
    ImGui::Text(OPTIONS_TEXT);
    updateDirectionalTextures |= ImGui::Checkbox("Log Scale", &doLogScale);
    ImGui::EndChild();

    // Reference Image Texture
    ImGui::SameLine(0.0f, paddingX.x);
    ImGui::BeginChild("refTexture", refImgSize, false);
    ImGui::SameLine(0.0f, GuideDebugGUIFuncs::CenteredTextLocation(sceneName.c_str(), refImgSize.x));
    ImGui::Text(sceneName.c_str());
    remainingSize = GuideDebugGUIFuncs::FindRemainingSize(refImgSize);
    remainingSize.x = remainingSize.y * (1.0f / 9.0f) * 16.0f;
    ImGui::NewLine();
    ImGui::SameLine(0.0f, (refImgSize.x - remainingSize.x) * 0.5f - ImGui::GetStyle().WindowPadding.x);

    ImVec2 refImgPos = ImGui::GetCursorScreenPos();

    {
        // Scope is here to use namespace since with structured bindings etc.
        // Statement become too long
        using namespace GuideDebugGUIFuncs;
        auto [pixChanged, newTexel] = RenderImageWithZoomTooltip(refTexture,
                                                                 worldPositions,
                                                                 remainingSize,
                                                                 pixelSelected,
                                                                 selectedPixel);

        if(pixChanged)
        {
            pixelSelected = true;
            updateDirectionalTextures = (selectedPixel != newTexel);
            selectedPixel = newTexel;
        }
    }
    ImGui::EndChild();

    // Before Render PG Textures Update the Directional Textures if requested
    if(updateDirectionalTextures && pixelSelected)
    {
        uint32_t linearIndex = (static_cast<uint32_t>(selectedPixel[1]) * refTexture.Size()[0] +
                                static_cast<uint32_t>(selectedPixel[0]));
        Vector3f worldPos = worldPositions[linearIndex];
        debugReference.UpdateDirectional(doLogScale,
                                         Vector2i(static_cast<int32_t>(selectedPixel[0]),
                                                  static_cast<int32_t>(selectedPixel[1])),
                                         Vector2i(static_cast<int32_t>(refTexture.Size()[0]),
                                                  static_cast<int32_t>(refTexture.Size()[1])));

        // Do Guide Debuggers
        for(const auto& renderer : debugRenderers)
        {
            renderer->UpdateDirectional(worldPos, doLogScale, currentDepth);
        }
        updateDirectionalTextures = false;
    }


    ImGui::SameLine(0.0f, paddingX.x);
    debugReference.RenderGUI(refPGImgSize);
    // New Line and Path Guider Images
    ImGui::Dummy(ImVec2(0.0f, std::max(0.0f, (paddingY - ImGui::GetFontSize()) * 0.95f)));

    ImGui::SameLine(0.0f, paddingX.y);
    for(const auto& renderer : debugRenderers)
    {
        if(renderer->RenderGUI(pgImgSize) && pixelSelected)
        {
            uint32_t linearIndex = (static_cast<uint32_t>(selectedPixel[1]) * refTexture.Size()[0] +
                                    static_cast<uint32_t>(selectedPixel[0]));
            Vector3f worldPos = worldPositions[linearIndex];
            renderer->UpdateDirectional(worldPos, doLogScale, currentDepth);
        }
        ImGui::SameLine(0.0f, paddingX.y);
    }
    // Finish Window
    ImGui::End();

    // Render the GUI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}