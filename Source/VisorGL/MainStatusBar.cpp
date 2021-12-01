#include "MainStatusBar.h"

#include <Imgui/imgui.h>
#include <Imgui/imgui_internal.h>

#include "RayLib/AnalyticData.h"

#include "IcoMoonFontTable.h"


namespace ImGui
{
    bool ToggleButton(const char* name, bool& toggle)
    {
        bool result = false;
        if(toggle == true)
        {
            ImVec4 hoverColor = ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered);

            ImGui::PushID(name);
            ImGui::PushStyleColor(ImGuiCol_Button, hoverColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, hoverColor);
            result = ImGui::Button(name);
            if(ImGui::IsItemClicked(0))
            {
                result = true;
                toggle = !toggle;
            }
            ImGui::PopStyleColor(2);
            ImGui::PopID();
        }
        else if(ImGui::Button(name))
        {
            result = true;
            toggle = true;
        }
        return result;
    }
}


MainStatusBar::MainStatusBar()
    : paused(false)
    , running(true)
    , stopped(false)
{}

void MainStatusBar::Render(const AnalyticData& ad,
                           const SceneAnalyticData& sad,
                           const Vector2i& iSize)
{
    using namespace std::string_literals;
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

            std::string prefix = std::string(RENDERING_NAME);
            std::string body = (prefix + " " + sad.sceneName + "...");
            if(paused)
                body += " ("s + std::string(PAUSED_NAME) + ")"s;
            else if(stopped)
                body += " ("s + std::string(STOPPED_NAME) + ")"s;
            ImGui::Text(body.c_str());

            float buttonSize = (ImGui::CalcTextSize(ICON_ICOMN_ARROW_LEFT).x +
                                ImGui::GetStyle().FramePadding.x * 2.0f);
            float spacingSize = ImGui::GetStyle().ItemSpacing.x;

            ImGui::SameLine(ImGui::GetWindowContentRegionMax().x -
                            (buttonSize * 5 +
                             spacingSize * 6 + 2));
            ImGui::Separator();
            ImGui::Button(ICON_ICOMN_ARROW_LEFT);
            if(ImGui::IsItemHovered() && GImGui->HoveredIdTimer > 1)
            {
                ImGui::BeginTooltip();
                ImGui::Text("Prev Frame");
                ImGui::EndTooltip();
            }

            ImGui::Button(ICON_ICOMN_ARROW_RIGHT);
            if(ImGui::IsItemHovered() && GImGui->HoveredIdTimer > 1)
            {
                ImGui::BeginTooltip();
                ImGui::Text("Next Frame");
                ImGui::EndTooltip();
            }
            ImGui::Separator();

            if(ImGui::ToggleButton(ICON_ICOMN_STOP2, stopped))
            {
                stopped = true;
                running = !stopped;
                paused = !stopped;
            }
            if(ImGui::ToggleButton(ICON_ICOMN_PAUSE2, paused))
            {
                if(!stopped)
                    running = !paused;
                else paused = false;
            }
            if(ImGui::ToggleButton(ICON_ICOMN_PLAY3, running))
            {
                running = true;
                stopped = !running;
                paused = !running;
            }
            ImGui::EndMenuBar();
        }
    }
    ImGui::End();
}


void MainStatusBar::SetState(RenderState rs)
{
    switch(rs)
    {
        case MainStatusBar::RUNNING:
        {
            stopped = paused = false;
            running = true;
            break;
        }
        case MainStatusBar::PAUSED:
        {
            running = stopped = false;
            paused = true;
            break;
        }
        case MainStatusBar::STOPPED:
        {
            running = paused = false;
            stopped = true;

            break;
        }
        default: break;
    }
}