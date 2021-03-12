#pragma once

struct GLFWwindow;

class VisorGUI
{
    private:
        static constexpr const char*    IMGUI_GLSL_STRING = "#version 430 core";
        bool init = false;

    protected:
    public:
        // Construtors & Destructor
                                        VisorGUI(GLFWwindow*);
                                        ~VisorGUI();
        // Members
        bool                            InitThread(GLFWwindow*);
        void                            RenderStart();
        void                            RenderEnd();

        void                            ProcessInputs();

        // Callbacks of the GUI

};