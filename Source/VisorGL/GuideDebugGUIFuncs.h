#pragma once

#include <type_traits>
#include <Imgui/imgui.h>
#include <Imgui/imgui_tex_inspect.h>

#include "RayLib/Vector.h"
#include "RayLib/HybridFunctions.h"

namespace GuideDebugGUIFuncs
{
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

    constexpr Vector2ui     PG_TEXTURE_SIZE = Vector2ui(512, 512);

    ImVec2      FindRemainingSize(const ImVec2& size);
    float       CenteredTextLocation(const char* text, float centeringWidth);
    template<class T>
    std::enable_if_t<std::is_same_v<T, Vector3f> ||
                     std::is_same_v<T, float>, std::tuple<bool, Vector2f>>
                RenderImageWithZoomTooltip(TextureGL&,
                                           const std::vector<T>& values,
                                           const ImVec2& size,
                                           bool renderCircle = false,
                                           const Vector2f& circleTexel = Zero2f);
}

template <class T>
GuideDebugGUIFuncs::ValueRenderer<T>::ValueRenderer(const std::vector<T>& values,
                                                    const Vector2ui& resolution)
    : resolution(resolution)
    , values(values)
{}

template <class T>
void GuideDebugGUIFuncs::ValueRenderer<T>::DrawAnnotation(ImDrawList* drawList, ImVec2 texel,
                                                          ImGuiTexInspect::Transform2D texelsToPixels, ImVec4 value)
{
    // Altered version of the tex inspector..
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
    // Interpolate alpha channel to gradually render the pixel values
    float ratioX = std::max(1.0f, std::abs(texelsToPixels.Scale.x) / textSize.x);
    ratioX = std::min(2.0f, ratioX);
    float ratioY = std::max(1.0f, std::abs(texelsToPixels.Scale.y) / textSize.y);
    ratioY = std::min(2.0f, ratioY);
    float ratio = std::min(ratioX, ratioY) - 1.0f;

    uint32_t alphaInt = static_cast<uint32_t>(ratio * 255.0f);
    alphaInt <<= 24;

    // Choose black or white text based on how bright the texel.  I.e. don't
    // draw black text on a dark background or vice versa.
    float brightness = (value.x + value.y + value.z) * value.w / 3;
    ImU32 lineColor = brightness > 0.5 ? 0x00000000 : 0x00FFFFFF;
    lineColor |= alphaInt;

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

    // Add text to draw list!
    ImVec2 pixelCenter = texelsToPixels * texel;
    pixelCenter.x -= textSize.x * 0.5f;
    pixelCenter.y -= textSize.y * 0.5f;

    drawList->AddText(pixelCenter, lineColor, worldPosAsString.c_str());
}

inline float GuideDebugGUIFuncs::CenteredTextLocation(const char* text, float centeringWidth)
{
    float widthText = ImGui::CalcTextSize(text).x;
    // Handle overflow
    if(widthText > centeringWidth) return 0;

    return (centeringWidth - widthText) * 0.5f;
}


inline ImVec2 GuideDebugGUIFuncs::FindRemainingSize(const ImVec2& size)
{
    return ImVec2(size.x - ImGui::GetCursorPos().x - ImGui::GetStyle().WindowPadding.x,
                  size.y - ImGui::GetCursorPos().y - ImGui::GetStyle().WindowPadding.y);
}

template<class T>
std::enable_if_t<std::is_same_v<T, Vector3f> ||
                 std::is_same_v<T, float>, std::tuple<bool, Vector2f>>
GuideDebugGUIFuncs::RenderImageWithZoomTooltip(TextureGL& tex,
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

    //flags |= ImGuiTexInspect::InspectorFlags_NoGrid;

    //ImVec2 imgStart = ImGui::GetCursorScreenPos();
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
            using namespace HybridFuncs;

            ImVec2 texel = transform * ImGui::GetMousePos();
            ImVec2 uv = ImVec2(texel.x / static_cast<float>(tex.Size()[0]),
                               texel.y / static_cast<float>(tex.Size()[1]));

            float texelClampedX = Clamp(texel[0], 0.0f,
                                        static_cast<float>(tex.Size()[0] - 1));
            float texelClampedY = Clamp(texel[1], 0.0f,
                                        static_cast<float>(tex.Size()[1] - 1));
            uint32_t linearIndex = (tex.Size()[0] * static_cast<uint32_t>(texelClampedY) +
                                                    static_cast<uint32_t>(texelClampedX));

            // Zoomed Tool-tip
            ImGui::BeginTooltip();
            static constexpr float ZOOM_FACTOR = 8.0f;
            static constexpr float REGION_SIZE = 16.0f;
            // Calculate Zoom UV
            float region_x = texel[0] - REGION_SIZE * 0.5f;
            region_x = Clamp(region_x, 0.0f, tex.Size()[0] - REGION_SIZE);
            float region_y = texel[1] - REGION_SIZE * 0.5f;
            region_y = Clamp(region_y, 0.0f, tex.Size()[1] - REGION_SIZE);

            ImVec2 zoomUVStart = ImVec2((region_x) / tex.Size()[0],
                                        (region_y) / tex.Size()[1]);
            ImVec2 zoomUVEnd = ImVec2((region_x + REGION_SIZE) / tex.Size()[0],
                                      (region_y + REGION_SIZE) / tex.Size()[1]);
            // Invert Y (.......)
            std::swap(zoomUVStart.y, zoomUVEnd.y);
            // Center the image on the tool-tip window
            ImVec2 ttImgSize(REGION_SIZE * ZOOM_FACTOR,
                             REGION_SIZE * ZOOM_FACTOR);
            ImGui::Image(texId, ttImgSize, zoomUVStart, zoomUVEnd);

            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::Text("Pixel: (%.2f, %.2f)", texel[0], texel[1]);
            ImGui::Text("UV   : (%.4f, %.4f)", uv[0], uv[1]);
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

            ImVec2 texelCenter = ImVec2(static_cast<uint32_t>(circleTexel[0]) + 0.5f,
                                        static_cast<uint32_t>(circleTexel[1]) + 0.5f);
            ImVec2 screenPixel = inverseT * ImVec2(texelCenter[0],
                                                   texelCenter[1]);

            //if(transform.Scale.x)

            ImDrawList* drawList = ImGui::GetWindowDrawList();
            drawList->AddCircle(screenPixel,
                                inverseT.Scale.x * 3.0f,
                                ImColor(0.0f, 1.0f, 0.0f), 0,
                                inverseT.Scale.x * 1.5f);
        }

    }
    ImGuiTexInspect::EndInspectorPanel();

    return result;
}