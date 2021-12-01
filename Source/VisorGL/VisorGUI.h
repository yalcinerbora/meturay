#pragma once

#include <GL/glew.h>

#include "TMOptionWindow.h"
#include "MainStatusBar.h"

#include "RayLib/Vector.h"
#include "RayLib/VisorCallbacksI.h"

struct GLFWwindow;
struct AnalyticData;
struct SceneAnalyticData;

class VisorGUI : public VisorCallbacksI
{
    private:
        static constexpr const char*    IMGUI_GLSL_STRING = "#version 430 core";

        TMOptionWindow                  tmWindow;
        MainStatusBar                   statusBar;

        bool                            topBarOn;
        bool                            bottomBarOn;

        VisorCallbacksI*                delegatedCallbacks;

    protected:
    public:
        // Construtors & Destructor
                                        VisorGUI(GLFWwindow*);
                                        ~VisorGUI();
        // Members
        void                            Render(const AnalyticData& ad,
                                               const SceneAnalyticData& sad,
                                               const Vector2i& resolution);
        // Access GUI Controlled Parameters
        const ToneMapOptions&           TMOptions() const;



        // Callbacks of the GUI
        // From Command Callbacks
        void                            ChangeScene(std::u8string) override;
        void                            ChangeTime(double) override;
        void                            IncreaseTime(double) override;
        void                            DecreaseTime(double) override;
        void                            ChangeCamera(VisorTransform) override;
        void                            ChangeCamera(unsigned int) override;
        void                            StartStopTrace(bool) override;
        void                            PauseContTrace(bool) override;

        void                            WindowMinimizeAction(bool minimized) override;
        void                            WindowCloseAction() override;
};

inline const ToneMapOptions& VisorGUI::TMOptions() const
{
    return tmWindow.TMOptions();
}