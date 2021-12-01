#pragma once

#include <map>
#include <GL/glew.h>
#include <glfw/glfw3.h>

#include "RayLib/VisorInputStructs.h"

class WindowInputI;

class GLFWCallbackDelegator
{
    public:
        static GLFWCallbackDelegator&           Instance();

    private:
        std::map<GLFWwindow*, WindowInputI*>    windowMappings;

        // METUray => GLFW converters
        static KeyAction                        DetermineAction(int);
        static MouseButtonType                  DetermineMouseButton(int);
        static KeyboardKeyType                  DetermineKey(int);

        // Callbacks
        // GLFW
        static void                             ErrorCallbackGLFW(int, const char*);
        static void                             WindowPosGLFW(GLFWwindow*, int, int);
        static void                             WindowFBGLFW(GLFWwindow*, int, int);
        static void                             WindowSizeGLFW(GLFWwindow*, int, int);
        static void                             WindowCloseGLFW(GLFWwindow*);
        static void                             WindowRefreshGLFW(GLFWwindow*);
        static void                             WindowFocusedGLFW(GLFWwindow*, int);
        static void                             WindowMinimizedGLFW(GLFWwindow*, int);

        static void                             KeyboardUsedGLFW(GLFWwindow*, int, int, int, int);
        static void                             MouseMovedGLFW(GLFWwindow*, double, double);
        static void                             MousePressedGLFW(GLFWwindow*, int, int, int);
        static void                             MouseScrolledGLFW(GLFWwindow*, double, double);

        // Constructors
                                                GLFWCallbackDelegator();
    protected:
    public:
        // Destructor
                                                GLFWCallbackDelegator(const GLFWCallbackDelegator&) = delete;
                                                GLFWCallbackDelegator(GLFWCallbackDelegator&&) = delete;
        GLFWCallbackDelegator&                  operator=(const GLFWCallbackDelegator&) = delete;
        GLFWCallbackDelegator&                  operator=(GLFWCallbackDelegator&&) = delete;
                                                ~GLFWCallbackDelegator();


        void                                    AttachWindow(GLFWwindow* glfwWindow, WindowInputI* window);
        void                                    DetachWindow(GLFWwindow* window);

        // OGL Debug write
        static void                             OGLDebugLog(GLenum type,
                                                            GLuint id,
                                                            GLenum severity,
                                                            const char* message);
};