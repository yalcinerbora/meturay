#pragma once

#include <GL/glew.h>

#include "TMOptionWindow.h"
#include "MainStatusBar.h"

#include "RayLib/Vector.h"

struct GLFWwindow;
struct AnalyticData;
struct SceneAnalyticData;

class VisorGUI
{
    private:
        static constexpr const char*    IMGUI_GLSL_STRING = "#version 430 core";
        bool init = false;

        TMOptionWindow                  tmWindow;
        MainStatusBar                   statusBar;

        bool                            topBarOn;
        bool                            bottomBarOn;

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
};

inline const ToneMapOptions& VisorGUI::TMOptions() const
{
    return tmWindow.TMOptions();
}