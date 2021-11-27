#pragma once

#include "TMOptionWindow.h"
#include <GL/glew.h>

#include "RayLib/Vector.h"

struct GLFWwindow;

class VisorGUI
{
    private:
        static constexpr const char*    IMGUI_GLSL_STRING = "#version 430 core";
        bool init = false;

        TMOptionWindow                  tmWindow;

        bool                            topBarOn;
        bool                            bottomBarOn;

    protected:
    public:
        // Construtors & Destructor
                                        VisorGUI(GLFWwindow*);
                                        ~VisorGUI();
        // Members
        void                            Render(GLuint sdrTex,
                                               const Vector2i& resolution);
        // Access GUI Controlled Parameters
        const ToneMapOptions&           TMOptions() const;

        // Callbacks of the GUI
};

inline const ToneMapOptions& VisorGUI::TMOptions() const
{
    return tmWindow.TMOptions();
}