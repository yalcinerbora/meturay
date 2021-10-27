#include "ImageIO/EntryPoint.h"

#include "GuideDebugGUI.h"

#include <Imgui/imgui.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_opengl3.h>

#include <Imgui/imgui_tex_inspect.h>
#include <Imgui/tex_inspect_opengl.h>

#include "RayLib/Log.h"
#include "RayLib/HybridFunctions.h"
#include "RayLib/VisorError.h"

#include "GDebugRendererReference.h"

template <class T>
class ValueRenderer
{
    protected:
        Vector2ui                       resolution;
        const std::vector<T>&           values;

        static constexpr int            TextColumnCount = 6;
        static constexpr int            TextRowCount = std::is_same_v<T, float>    ? 1 :
                                                       std::is_same_v<T, Vector2f> ? 2 :
                                                       std::is_same_v<T, Vector3f> ? 3 :
                                                       std::is_same_v<T, Vector4f> ? 4 :
                                                       std::numeric_limits<uint32_t>::max();

    public:
        // Constructors & Destructor
                        ValueRenderer(const std::vector<T>& , const Vector2ui& resolution);
                        ~ValueRenderer() = default;

    void                DrawAnnotation(ImDrawList* drawList, ImVec2 texel,
                                       ImGuiTexInspect::Transform2D texelsToPixels, ImVec4 value);
};

template <class T>
ValueRenderer<T>::ValueRenderer(const std::vector<T>& values, const Vector2ui& resolution)
    : resolution(resolution)
    , values(values)
{}

template <class T>
void ValueRenderer<T>::DrawAnnotation(ImDrawList* drawList, ImVec2 texel,
                                      ImGuiTexInspect::Transform2D texelsToPixels, ImVec4 value)
{

    std::string worldPosAsString;

    float fontHeight = ImGui::GetFontSize();
    // WARNING this is a hack that gets a constant
    // character width from half the height.  This work for the default font but
    // won't work on other fonts which may even not be monospace.
    float fontWidth = fontHeight / 2;

    // Calculate size of text and check if it fits
    ImVec2 textSize = ImVec2((float)TextColumnCount * fontWidth,
                             (float)TextRowCount * fontHeight);

    if (textSize.x > std::abs(texelsToPixels.Scale.x) ||
        textSize.y > std::abs(texelsToPixels.Scale.y))
    {
        // Not enough room in texel to fit the text.  Don't draw it.
        return;
    }
    /* Choose black or white text based on how bright the texel.  I.e. don't
     * draw black text on a dark background or vice versa. */
    float brightness = (value.x + value.y + value.z) * value.w / 3;
    ImU32 lineColor = brightness > 0.5 ? 0xFF000000 : 0xFFFFFFFF;

    uint32_t linearId = static_cast<uint32_t>(texel.y) * resolution[0] +
                        static_cast<uint32_t>(texel.x);
    T dispValue = values[linearId];

    static constexpr std::string_view format = std::is_same_v<T, float>    ? "{:5.3f}" :
                                               std::is_same_v<T, Vector2f> ? "{:5.3f}\n{:5.3f}" :
                                               std::is_same_v<T, Vector3f> ? "{:5.3f}\n{:5.3f}\n{:5.3f}" :
                                               std::is_same_v<T, Vector4f> ? "{:5.3f}\n{:5.3f}\n{:5.3f}\n{:5.3f}" :
                                               "";
    if constexpr (std::is_same_v<T, float>)
        worldPosAsString = fmt::format(format, dispValue);
    else if constexpr(std::is_same_v<T, Vector2f>)
        worldPosAsString = fmt::format(format, dispValue[0], dispValue[1]);
    else if constexpr(std::is_same_v<T, Vector3f>)
        worldPosAsString = fmt::format(format, dispValue[0], dispValue[1], dispValue[2]);
    else if constexpr (std::is_same_v<T, Vector4f>)
        worldPosAsString = fmt::format(format, dispValue[0], dispValue[1], dispValue[2], dispValue[3]);
    else
        worldPosAsString = "INV-VAL";

    // Add text to drawlist!
    ImVec2 pixelCenter = texelsToPixels * texel;
    pixelCenter.x -= textSize.x * 0.5f;
    pixelCenter.y -= textSize.y * 0.5f;

    drawList->AddText(pixelCenter, lineColor, worldPosAsString.c_str());
}

float GuideDebugGUI::CenteredTextLocation(const char* text, float centeringWidth)
{
    float widthText = ImGui::CalcTextSize(text).x;
    // Handle overflow
    if(widthText > centeringWidth) return 0;

    return (centeringWidth - widthText) * 0.5f;
}

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

template<class T>
std::enable_if_t<std::is_same_v<T, Vector3f> ||
                 std::is_same_v<T, float>, std::tuple<bool, Vector2f>>
GuideDebugGUI::RenderImageWithZoomTooltip(TextureGL& tex,
                                          const std::vector<T>& values,
                                          const ImVec2& size,
                                          bool renderCircle,
                                          const Vector2f& circleTexel)
{
    auto result = std::make_tuple(false, Zero2f);

    // Debug Reference Image
    ImTextureID texId = (void*)(intptr_t)tex.TexId();
    ImGuiTexInspect::InspectorFlags flags = 0;
    flags |= ImGuiTexInspect::InspectorFlags_FlipY;
    flags |= ImGuiTexInspect::InspectorFlags_NoZoomOut;
    flags |= ImGuiTexInspect::InspectorFlags_FillVertical;
    flags |= ImGuiTexInspect::InspectorFlags_NoAutoReadTexture;
    flags |= ImGuiTexInspect::InspectorFlags_NoBorder;
    flags |= ImGuiTexInspect::InspectorFlags_NoTooltip;

    flags |= ImGuiTexInspect::InspectorFlags_NoGrid;

    ImVec2 imgStart = ImGui::GetCursorScreenPos();
    if(ImGuiTexInspect::BeginInspectorPanel("##RefImage", texId,
                                            ImVec2(static_cast<float>(tex.Size()[0]),
                                                   static_cast<float>(tex.Size()[1])),
                                            flags,
                                            ImGuiTexInspect::SizeIncludingBorder(size)))
    {
        // Draw some text showing color value of each texel (you must be zoomed in to see this)
        ImGuiTexInspect::DrawAnnotations(ValueRenderer(values, tex.Size()));
        ImGuiTexInspect::Transform2D transform = ImGuiTexInspect::CurrentInspector_GetTransform();

        if(ImGui::IsItemHovered() && tex.Size() != Zero2ui)
        {
            ImVec2 texel = transform * ImGui::GetMousePos();
            uint32_t linearIndex = (tex.Size()[0] * static_cast<uint32_t>(texel[1]) +
                                                    static_cast<uint32_t>(texel[0]));

            // Zoomed Tooltip
            ImGui::BeginTooltip();

            static constexpr float ZOOM_FACTOR = 8.0f;
            static constexpr float REGION_SIZE = 16.0f;
            // Calculate Zoom UV
            float region_x = texel[0] - REGION_SIZE * 0.5f;
            region_x = HybridFuncs::Clamp(region_x, 0.0f, tex.Size()[0] - REGION_SIZE);
            float region_y = texel[1] - REGION_SIZE * 0.5f;
            region_y = HybridFuncs::Clamp(region_y, 0.0f, tex.Size()[1] - REGION_SIZE);

            ImVec2 uv0 = ImVec2((region_x) / tex.Size()[0],
                                (region_y) / tex.Size()[1]);
            ImVec2 uv1 = ImVec2((region_x + REGION_SIZE) / tex.Size()[0],
                                (region_y + REGION_SIZE) / tex.Size()[1]);
            // Invert Y (.......)
            std::swap(uv0.y, uv1.y);
            // Center the image on the tooltip window
            ImVec2 ttImgSize(REGION_SIZE * ZOOM_FACTOR,
                             REGION_SIZE * ZOOM_FACTOR);
            ImGui::Image(texId, ttImgSize, uv0, uv1);

            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::Text("Pixel: (%.2f, %.2f)", texel[0], texel[1]);
            if constexpr(std::is_same_v<T, float>)
                ImGui::Text("Value: %f", values[linearIndex]);
            else if constexpr(std::is_same_v<T, Vector3f>)
                ImGui::TextWrapped("WorldPos: %f\n"
                                   "          %f\n"
                                   "          %f",
                            values[linearIndex][0],
                            values[linearIndex][1],
                            values[linearIndex][2]);
            ImGui::EndGroup();
            ImGui::EndTooltip();

            // Render circle on the clicked pos if requested
            if(ImGui::IsMouseClicked(ImGuiMouseButton_::ImGuiMouseButton_Left))
                result = std::make_tuple(true, Vector2f(texel[0], texel[1]));
        }

        // Draw a circle on the selected location
        if(renderCircle)
        {
            ImGuiTexInspect::Transform2D inverseT = transform.Inverse();

            ImVec2 screenPixel = inverseT * ImVec2(circleTexel[0],
                                                   circleTexel[1]);

            ImDrawList* drawList = ImGui::GetWindowDrawList();
            drawList->AddCircle(screenPixel,
                                (1.0f / transform.Scale.x) * 3.0f,
                                ImColor(0.0f, 1.0f, 0.0f), 0, 1.5f);
        }

    }
    ImGuiTexInspect::EndInspectorPanel();

    return result;
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

ImVec2 GuideDebugGUI::FindRemainingSize(const ImVec2& size)
{
    return ImVec2(size.x - ImGui::GetCursorPos().x - ImGui::GetStyle().WindowPadding.x,
                  size.y - ImGui::GetCursorPos().y - ImGui::GetStyle().WindowPadding.y);
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
    ImGui::SameLine(0.0f, CenteredTextLocation(OPTIONS_TEXT, optionsSize.x));
    ImGui::Text(OPTIONS_TEXT);
    updateDirectionalTextures |= ImGui::Checkbox("Log Scale", &doLogScale);
    ImGui::EndChild();

    // Reference Image Texture
    ImGui::SameLine(0.0f, paddingX.x);
    //ImVec2 childStart = ImGui::GetCursorPos
    ImGui::BeginChild("refTexture", refImgSize, true);
    ImGui::SameLine(0.0f, CenteredTextLocation(sceneName.c_str(), refImgSize.x));
    ImGui::Text(sceneName.c_str());
    remainingSize = FindRemainingSize(refImgSize);
    remainingSize.x = remainingSize.y * (1.0f / 9.0f) * 16.0f;
    ImGui::NewLine();
    ImGui::SameLine(0.0f, (refImgSize.x - remainingSize.x) * 0.5f - ImGui::GetStyle().WindowPadding.x);

    ImVec2 refImgPos = ImGui::GetCursorScreenPos();
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
    ImGui::EndChild();

    // Before Render PG Textures Update the Directional Textures if requested
    if(updateDirectionalTextures && pixelSelected)
    {
        uint32_t linearIndex = (static_cast<uint32_t>(selectedPixel[1]) * refTexture.Size()[0] +
                                static_cast<uint32_t>(selectedPixel[0]));
        Vector3f worldPos = worldPositions[linearIndex];
        debugReference.RenderDirectional(debugRefTexture, debugRefPixValues, doLogScale,
                                         Vector2i(static_cast<int32_t>(selectedPixel[0]),
                                                  static_cast<int32_t>(selectedPixel[1])),
                                         Vector2i(static_cast<int32_t>(refTexture.Size()[0]),
                                                  static_cast<int32_t>(refTexture.Size()[1])));

        // Do Guide Debuggers
        for(size_t i = 0; i < guideTextues.size(); i++)
        {
            debugRenderers[i]->RenderDirectional(guideTextues[i],
                                                 guidePixValues[i],
                                                 worldPos, doLogScale,
                                                 currentDepth);
        }
        updateDirectionalTextures = false;
    }



    ImGui::SameLine(0.0f, paddingX.x);
    ImGui::BeginChild("debugTexture", refPGImgSize, true);
    ImGui::SameLine(0.0f, CenteredTextLocation(REFERENCE_TEXT, refPGImgSize.x));
    ImGui::Text(REFERENCE_TEXT);
    remainingSize = FindRemainingSize(refPGImgSize);
    remainingSize.x = remainingSize.y;
    ImGui::NewLine();
    ImGui::SameLine(0.0f, (refPGImgSize.x - remainingSize.x) * 0.5f - ImGui::GetStyle().WindowPadding.x);
    // Debug Reference Image
    if(pixelSelected)
        RenderImageWithZoomTooltip(debugRefTexture, debugRefPixValues, remainingSize);
    else
        ImGui::Dummy(remainingSize);
    ImGui::EndChild();

    // New Line and Path Guider Images
    ImGui::Dummy(ImVec2(0.0f, std::max(0.0f, (paddingY - ImGui::GetFontSize()) * 0.95f)));

    ImGui::SetCursorPosX(paddingX.y);
    for(size_t i = 0; i < guideTextues.size(); i++)
    {
        ImGui::BeginChild(("debugTexture" + std::to_string(i)).c_str(), pgImgSize, true);
        ImGui::SameLine(0.0f, CenteredTextLocation(debugRenderers[i]->Name().c_str(), pgImgSize.x));
        ImGui::Text(debugRenderers[i]->Name().c_str());
        remainingSize = FindRemainingSize(pgImgSize);
        remainingSize.x = remainingSize.y;
        ImGui::NewLine();
        ImGui::SameLine(0.0f, (pgImgSize.x - remainingSize.x) * 0.5f - ImGui::GetStyle().WindowPadding.x);
        RenderImageWithZoomTooltip(guideTextues[i], guidePixValues[i], remainingSize);

        //if(ImGui::BeginPopupContextItem())
        //{
        //    ImGui::Button("ABC");
        //}
        //ImGui::EndPopup();

        //ImGui::EndChild();

        ImGui::SameLine(0.0f, paddingX.y);
    }
    // Finish Window
    ImGui::End();

    // Render the GUI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}