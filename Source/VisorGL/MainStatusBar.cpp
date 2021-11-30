#include "MainStatusBar.h"

#include <Imgui/imgui.h>
#include <Imgui/imgui_internal.h>

#include "RayLib/AnalyticData.h"

#include "IcoMoonFontTable.h"

void MainStatusBar::Render(const AnalyticData& ad,
                           const SceneAnalyticData& sad,
                           const Vector2i& iSize)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar |
                                    ImGuiWindowFlags_NoSavedSettings |
                                    ImGuiWindowFlags_MenuBar;
    float height = ImGui::GetFrameHeight();
    if(ImGui::BeginViewportSideBar("##MainStatusBar", NULL, ImGuiDir_Down, height, window_flags))
    {
        if(ImGui::BeginMenuBar())
        {
            ImGui::Text((std::to_string(iSize[0]) + "x" + std::to_string(iSize[1])).c_str());
            ImGui::Separator();
            ImGui::Text(fmt::format("{:>7.3f}{:s}", ad.throughput, ad.throughputSuffix).c_str());
            ImGui::Separator();
            ImGui::Text((fmt::format("{:>6.0f}{:s}", ad.workPerPixel, ad.workPerPixelSuffix).c_str()));
            ImGui::Separator();
            ImGui::Text((std::string(RENDERING) + " " + sad.sceneName + "...").c_str());


            float buttonSize = (ImGui::CalcTextSize(ICON_ICOMN_ARROW_LEFT).x +
                                ImGui::GetStyle().FramePadding.x * 2.0f);
            float spacingSize = ImGui::GetStyle().ItemSpacing.x;

            ImGui::SameLine(ImGui::GetWindowContentRegionMax().x -
                            (buttonSize * 5 +
                             spacingSize * 6 + 2));
            ImGui::Separator();
            ImGui::Button(ICON_ICOMN_ARROW_LEFT);
            if(ImGui::IsItemHovered() && GImGui->HoveredIdTimer > 2)
            {
                ImGui::BeginTooltip();
                ImGui::Text("Prev Frame");
                ImGui::EndTooltip();
            }

            ImGui::Button(ICON_ICOMN_ARROW_RIGHT);
            if(ImGui::IsItemHovered() && GImGui->HoveredIdTimer > 2)
            {
                ImGui::BeginTooltip();
                ImGui::Text("Next Frame");
                ImGui::EndTooltip();
            }
            ImGui::Separator();
            ImGui::Button(ICON_ICOMN_STOP2);
            ImGui::Button(ICON_ICOMN_PAUSE2);
            ImGui::Button(ICON_ICOMN_PLAY3);




            ImGui::EndMenuBar();
        }
    }
    ImGui::End();
}