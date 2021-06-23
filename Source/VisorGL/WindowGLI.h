#pragma once


#include "RayLib/VisorI.h"

class GLFWCallbackDelegator;

class WindowGLI : public VisorI
{
    private:        
        virtual void            SetFBSizeFromInput(const Vector2i&) = 0;
        virtual void            SetWindowSizeFromInput(const Vector2i&) = 0;
        virtual void            SetOpenStateFromInput(bool) = 0;
        virtual VisorInputI*    InputInterface() = 0;

        // This is here just to hide these functions from user
        friend class            GLFWCallbackDelegator;

    public:
        virtual                 ~WindowGLI() = default;
};