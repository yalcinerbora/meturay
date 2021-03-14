#pragma once

#include "TMOptionWindow.h"

struct GLFWwindow;



class VisorGUI
{
    private:
        static constexpr const char*    IMGUI_GLSL_STRING = "#version 430 core";
        bool init = false;

        TMOptionWindow                  tmWindow;

    protected:
    public:
        // Construtors & Destructor
                                        VisorGUI(GLFWwindow*);
                                        ~VisorGUI();
        // Members
        void                            Render();
        // Access GUI Controlled Parameters
        const ToneMapOptions&           ToneMapOptions() const;

        // Callbacks of the GUI



};

inline const ToneMapOptions& VisorGUI::ToneMapOptions() const
{
    return tmWindow.GetToneMapOptions();
}