#pragma once

#include <GL/glew.h>

#include "TMOptionWindow.h"
#include "MainStatusBar.h"
#include "VisorWindowInput.h"

#include "RayLib/Vector.h"
#include "RayLib/VisorCallbacksI.h"
#include "RayLib/AnalyticData.h"
#include "RayLib/TracerOptions.h"

struct GLFWwindow;

class VisorGUI : public VisorWindowInput
{
    private:
        static constexpr const char*    IMGUI_GLSL_STRING = "#version 430 core";

        TMOptionWindow          tmWindow;
        MainStatusBar           statusBar;

        bool                    topBarOn;
        bool                    bottomBarOn;

        TracerAnalyticData      tracerAnalyticData;
        SceneAnalyticData       sceneAnalyticData;
        TracerOptions           currentTOpts;
        TracerParameters        currentTParams;

        // Read Only State
        const Vector2i&         imageSize;

    protected:
    public:
        // Constructors & Destructor
                        VisorGUI(VisorCallbacksI&,
                                 bool& isWindowOpen,
                                 Vector2i& windowSize,
                                 Vector2i& viewportSize,
                                 ToneMapOptions& tmOpts,
                                 ImageSaverI& saver,
                                 const Vector2i& imageSize,
                                 const GLFWwindow*,
                                 const KeyboardKeyBindings&,
                                 const MouseKeyBindings&,
                                 MovementSchemeList&&);
        virtual         ~VisorGUI();

        // Members
        void            RenderGUI() override;

        void            SetSceneAnalyticData(const SceneAnalyticData&) override;
        void            SetTracerAnalyticData(const TracerAnalyticData&) override;
        void            SetTracerOptions(const TracerOptions&) override;
        void            SetTracerParams(const TracerParameters&) override;

};