#include "TMOptionWindow.h"

#include "RayLib/Log.h"

#include <Imgui/imgui.h>

void TMOptionWindow::Render()
{
    if(!windowOpen) return;

    // TODO: Implement
    if(ImGui::Begin("Tone Map Options", &windowOpen,
                    ImGuiWindowFlags_NoCollapse))
    {
        // TODO: Align these properly
        ImGui::Checkbox("Tone Map On  ", &opts.doToneMap);
        ImGui::Checkbox("Gamma On     ", &opts.doGamma);
        ImGui::Checkbox("Key Adjust On", &opts.doKeyAdjust);
        // Gamma
        ImGui::Text("Gamma     ");
        ImGui::SameLine();
        ImGui::SliderFloat("##GammaSlider", &opts.gamma, 0.1f, 4.0f, "%0.3f");
        // Burn Ratio
        ImGui::Text("Burn Ratio");
        ImGui::SameLine();
        ImGui::SliderFloat("##BurnRatioSlider", &opts.burnRatio, 0.5f, 2.0f);
        // Key
        ImGui::Text("Key       ");
        ImGui::SameLine();
        ImGui::InputFloat("##Key", &opts.key, 0.01f, 0.1f);
    }
    ImGui::End();
}