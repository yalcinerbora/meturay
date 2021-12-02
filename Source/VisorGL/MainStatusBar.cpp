#include "MainStatusBar.h"

#include <Imgui/imgui.h>
#include <Imgui/imgui_internal.h>

#include "RayLib/AnalyticData.h"
#include "RayLib/TracerStructs.h"
#include "RayLib/VisorCallbacksI.h"

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

void MainStatusBar::Render(VisorCallbacksI& cb,
                           TracerRunState& rs,
                           const TracerAnalyticData& ad,
                           const SceneAnalyticData& sad,
                           const Vector2i& iSize)
{
    using namespace std::string_literals;
    constexpr ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar |
                                              ImGuiWindowFlags_NoSavedSettings |
                                              ImGuiWindowFlags_MenuBar;

    // Pre-init button state if it changed by keyboard
    SetButtonState(stopped, running, paused, rs);

    float height = ImGui::GetFrameHeight();
    if(ImGui::BeginViewportSideBar("##MainStatusBar", NULL, ImGuiDir_Down, height, window_flags))
    {
        if(ImGui::BeginMenuBar())
        {
            double usedGPUMemMiB = ad.usedGPUMemoryMiB;
            double totalGPUMemGiB = ad.totalGPUMemoryMiB / 1024.0;
            std::string memUsage = fmt::format("{:.1f}MiB / {:.1f}GiB",
                                               usedGPUMemMiB, totalGPUMemGiB);

            ImGui::Text("%s", memUsage.c_str());
            ImGui::Separator();
            ImGui::Text("%s", (std::to_string(iSize[0]) + "x" + std::to_string(iSize[1])).c_str());
            ImGui::Separator();
            ImGui::Text("%s", fmt::format("{:>7.3f}{:s}", ad.throughput, ad.throughputSuffix).c_str());
            ImGui::Separator();
            ImGui::Text("%s", (fmt::format("{:>6.0f}{:s}", ad.workPerPixel, ad.workPerPixelSuffix).c_str()));
            ImGui::Separator();

            std::string prefix = std::string(RENDERING_NAME);
            std::string body = (prefix + " " + sad.sceneName + "...");
            if(paused)
                body += " ("s + std::string(PAUSED_NAME) + ")"s;
            else if(stopped)
                body += " ("s + std::string(STOPPED_NAME) + ")"s;
            ImGui::Text("%s", body.c_str());

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
                if(running || paused)
                    cb.StartStopTrace(false);

                stopped = true;
                running = !stopped;
                paused = !stopped;

            }
            if(ImGui::ToggleButton(ICON_ICOMN_PAUSE2, paused))
            {
                if(!stopped)
                {
                    cb.PauseContTrace(paused);
                    running = !paused;
                }
                else paused = false;
            }
            if(ImGui::ToggleButton(ICON_ICOMN_PLAY3, running))
            {
                if(stopped) cb.StartStopTrace(true);
                else if(paused) cb.PauseContTrace(false);

                running = true;
                stopped = false;
                paused = false;
            }
            ImGui::EndMenuBar();
        }
    }
    ImGui::End();

    rs = DetermineTracerState(stopped, running, paused);
}

TracerRunState MainStatusBar::DetermineTracerState(bool stopToggle,
                                                   bool,
                                                   bool pauseToggle)
{
    if(stopToggle)
        return TracerRunState::STOPPED;
    else if(pauseToggle)
        return TracerRunState::PAUSED;
    return TracerRunState::RUNNING;
}

void MainStatusBar::SetButtonState(bool& stopToggle,
                                   bool& runToggle,
                                   bool& pauseToggle,
                                   TracerRunState tracerRunState)
{
    stopToggle = false;
    runToggle = false;
    pauseToggle = false;
    switch(tracerRunState)
    {
        case TracerRunState::PAUSED:
            pauseToggle = true; break;
        case TracerRunState::RUNNING:
            runToggle = true; break;
        case TracerRunState::STOPPED:
            stopToggle = true; break;
        default: break;
    }
}