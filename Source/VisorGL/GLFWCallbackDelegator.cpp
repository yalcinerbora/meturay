#include "GLFWCallbackDelegator.h"

#include "RayLib/Log.h"
#include "RayLib/VisorError.h"

#include "VisorGL.h"

GLFWCallbackDelegator& GLFWCallbackDelegator::Instance()
{
    static GLFWCallbackDelegator ins;
    return ins;
}

KeyAction GLFWCallbackDelegator::DetermineAction(int action)
{
    if(action == GLFW_PRESS)
    {
        return KeyAction::PRESSED;
    }
    else if(action == GLFW_RELEASE)
    {
        return KeyAction::RELEASED;
    }
    else if(action == GLFW_REPEAT)
    {
        return KeyAction::REPEATED;
    }
    else
    {
        assert(false);
        return KeyAction::PRESSED;
    }
}

MouseButtonType GLFWCallbackDelegator::DetermineMouseButton(int button)
{
    static std::map<int, MouseButtonType> buttonMap =
    {
        std::make_pair(GLFW_MOUSE_BUTTON_LEFT, MouseButtonType::LEFT),
        std::make_pair(GLFW_MOUSE_BUTTON_RIGHT, MouseButtonType::RIGHT),
        std::make_pair(GLFW_MOUSE_BUTTON_MIDDLE, MouseButtonType::MIDDLE),
        std::make_pair(GLFW_MOUSE_BUTTON_4, MouseButtonType::BUTTON_4),
        std::make_pair(GLFW_MOUSE_BUTTON_5, MouseButtonType::BUTTON_5),
        std::make_pair(GLFW_MOUSE_BUTTON_6, MouseButtonType::BUTTON_6),
        std::make_pair(GLFW_MOUSE_BUTTON_7, MouseButtonType::BUTTON_7),
        std::make_pair(GLFW_MOUSE_BUTTON_8, MouseButtonType::BUTTON_8)
    };
    return buttonMap[button];
}

KeyboardKeyType GLFWCallbackDelegator::DetermineKey(int key)
{
    static std::map<int, KeyboardKeyType> keyMap =
    {
        std::make_pair(GLFW_KEY_SPACE, KeyboardKeyType::SPACE),
        std::make_pair(GLFW_KEY_APOSTROPHE, KeyboardKeyType::APOSTROPHE),
        std::make_pair(GLFW_KEY_COMMA, KeyboardKeyType::COMMA),
        std::make_pair(GLFW_KEY_MINUS, KeyboardKeyType::MINUS),
        std::make_pair(GLFW_KEY_PERIOD, KeyboardKeyType::PERIOD),
        std::make_pair(GLFW_KEY_SLASH, KeyboardKeyType::SLASH),
        std::make_pair(GLFW_KEY_0, KeyboardKeyType::NUMBER_0),
        std::make_pair(GLFW_KEY_1, KeyboardKeyType::NUMBER_1),
        std::make_pair(GLFW_KEY_2, KeyboardKeyType::NUMBER_2),
        std::make_pair(GLFW_KEY_3, KeyboardKeyType::NUMBER_3),
        std::make_pair(GLFW_KEY_4, KeyboardKeyType::NUMBER_4),
        std::make_pair(GLFW_KEY_5, KeyboardKeyType::NUMBER_5),
        std::make_pair(GLFW_KEY_6, KeyboardKeyType::NUMBER_6),
        std::make_pair(GLFW_KEY_7, KeyboardKeyType::NUMBER_7),
        std::make_pair(GLFW_KEY_8, KeyboardKeyType::NUMBER_8),
        std::make_pair(GLFW_KEY_9, KeyboardKeyType::NUMBER_9),
        std::make_pair(GLFW_KEY_SEMICOLON, KeyboardKeyType::SEMICOLON),
        std::make_pair(GLFW_KEY_EQUAL, KeyboardKeyType::EQUAL),
        std::make_pair(GLFW_KEY_A, KeyboardKeyType::A),
        std::make_pair(GLFW_KEY_B, KeyboardKeyType::B),
        std::make_pair(GLFW_KEY_C, KeyboardKeyType::C),
        std::make_pair(GLFW_KEY_D, KeyboardKeyType::D),
        std::make_pair(GLFW_KEY_E, KeyboardKeyType::E),
        std::make_pair(GLFW_KEY_F, KeyboardKeyType::F),
        std::make_pair(GLFW_KEY_G, KeyboardKeyType::G),
        std::make_pair(GLFW_KEY_H, KeyboardKeyType::H),
        std::make_pair(GLFW_KEY_I, KeyboardKeyType::I),
        std::make_pair(GLFW_KEY_J, KeyboardKeyType::J),
        std::make_pair(GLFW_KEY_K, KeyboardKeyType::K),
        std::make_pair(GLFW_KEY_L, KeyboardKeyType::L),
        std::make_pair(GLFW_KEY_M, KeyboardKeyType::M),
        std::make_pair(GLFW_KEY_N, KeyboardKeyType::N),
        std::make_pair(GLFW_KEY_O, KeyboardKeyType::O),
        std::make_pair(GLFW_KEY_P, KeyboardKeyType::P),
        std::make_pair(GLFW_KEY_Q, KeyboardKeyType::Q),
        std::make_pair(GLFW_KEY_R, KeyboardKeyType::R),
        std::make_pair(GLFW_KEY_S, KeyboardKeyType::S),
        std::make_pair(GLFW_KEY_T, KeyboardKeyType::T),
        std::make_pair(GLFW_KEY_U, KeyboardKeyType::U),
        std::make_pair(GLFW_KEY_V, KeyboardKeyType::V),
        std::make_pair(GLFW_KEY_W, KeyboardKeyType::W),
        std::make_pair(GLFW_KEY_X, KeyboardKeyType::X),
        std::make_pair(GLFW_KEY_Y, KeyboardKeyType::Y),
        std::make_pair(GLFW_KEY_Z, KeyboardKeyType::Z),
        std::make_pair(GLFW_KEY_LEFT_BRACKET, KeyboardKeyType::LEFT_BRACKET),
        std::make_pair(GLFW_KEY_BACKSLASH, KeyboardKeyType::BACKSLASH),
        std::make_pair(GLFW_KEY_RIGHT_BRACKET, KeyboardKeyType::RIGHT_BRACKET),
        std::make_pair(GLFW_KEY_GRAVE_ACCENT, KeyboardKeyType::GRAVE_ACCENT),
        std::make_pair(GLFW_KEY_WORLD_1, KeyboardKeyType::WORLD_1),
        std::make_pair(GLFW_KEY_WORLD_2, KeyboardKeyType::WORLD_2),
        std::make_pair(GLFW_KEY_ESCAPE, KeyboardKeyType::ESCAPE),
        std::make_pair(GLFW_KEY_ENTER, KeyboardKeyType::ENTER),
        std::make_pair(GLFW_KEY_TAB, KeyboardKeyType::TAB),
        std::make_pair(GLFW_KEY_BACKSPACE, KeyboardKeyType::BACKSPACE),
        std::make_pair(GLFW_KEY_INSERT, KeyboardKeyType::INSERT),
        std::make_pair(GLFW_KEY_DELETE, KeyboardKeyType::DELETE_KEY),
        std::make_pair(GLFW_KEY_RIGHT, KeyboardKeyType::RIGHT),
        std::make_pair(GLFW_KEY_LEFT, KeyboardKeyType::LEFT),
        std::make_pair(GLFW_KEY_DOWN, KeyboardKeyType::DOWN),
        std::make_pair(GLFW_KEY_UP, KeyboardKeyType::UP),
        std::make_pair(GLFW_KEY_PAGE_UP, KeyboardKeyType::PAGE_UP),
        std::make_pair(GLFW_KEY_PAGE_DOWN, KeyboardKeyType::PAGE_DOWN),
        std::make_pair(GLFW_KEY_HOME, KeyboardKeyType::HOME),
        std::make_pair(GLFW_KEY_END, KeyboardKeyType::END),
        std::make_pair(GLFW_KEY_CAPS_LOCK, KeyboardKeyType::CAPS_LOCK),
        std::make_pair(GLFW_KEY_SCROLL_LOCK, KeyboardKeyType::SCROLL_LOCK),
        std::make_pair(GLFW_KEY_NUM_LOCK, KeyboardKeyType::NUM_LOCK),
        std::make_pair(GLFW_KEY_PRINT_SCREEN, KeyboardKeyType::PRINT_SCREEN),
        std::make_pair(GLFW_KEY_PAUSE, KeyboardKeyType::PAUSE),
        std::make_pair(GLFW_KEY_F1, KeyboardKeyType::F1),
        std::make_pair(GLFW_KEY_F2, KeyboardKeyType::F2),
        std::make_pair(GLFW_KEY_F3, KeyboardKeyType::F3),
        std::make_pair(GLFW_KEY_F4, KeyboardKeyType::F4),
        std::make_pair(GLFW_KEY_F5, KeyboardKeyType::F5),
        std::make_pair(GLFW_KEY_F6, KeyboardKeyType::F6),
        std::make_pair(GLFW_KEY_F7, KeyboardKeyType::F7),
        std::make_pair(GLFW_KEY_F8, KeyboardKeyType::F8),
        std::make_pair(GLFW_KEY_F9, KeyboardKeyType::F9),
        std::make_pair(GLFW_KEY_F10, KeyboardKeyType::F10),
        std::make_pair(GLFW_KEY_F11, KeyboardKeyType::F11),
        std::make_pair(GLFW_KEY_F12, KeyboardKeyType::F12),
        std::make_pair(GLFW_KEY_F13, KeyboardKeyType::F13),
        std::make_pair(GLFW_KEY_F14, KeyboardKeyType::F14),
        std::make_pair(GLFW_KEY_F15, KeyboardKeyType::F15),
        std::make_pair(GLFW_KEY_F16, KeyboardKeyType::F16),
        std::make_pair(GLFW_KEY_F17, KeyboardKeyType::F17),
        std::make_pair(GLFW_KEY_F18, KeyboardKeyType::F18),
        std::make_pair(GLFW_KEY_F19, KeyboardKeyType::F19),
        std::make_pair(GLFW_KEY_F20, KeyboardKeyType::F20),
        std::make_pair(GLFW_KEY_F21, KeyboardKeyType::F21),
        std::make_pair(GLFW_KEY_F22, KeyboardKeyType::F22),
        std::make_pair(GLFW_KEY_F23, KeyboardKeyType::F23),
        std::make_pair(GLFW_KEY_F24, KeyboardKeyType::F24),
        std::make_pair(GLFW_KEY_F25, KeyboardKeyType::F25),
        std::make_pair(GLFW_KEY_KP_0, KeyboardKeyType::KP_0),
        std::make_pair(GLFW_KEY_KP_1, KeyboardKeyType::KP_1),
        std::make_pair(GLFW_KEY_KP_2, KeyboardKeyType::KP_2),
        std::make_pair(GLFW_KEY_KP_3, KeyboardKeyType::KP_3),
        std::make_pair(GLFW_KEY_KP_4, KeyboardKeyType::KP_4),
        std::make_pair(GLFW_KEY_KP_5, KeyboardKeyType::KP_5),
        std::make_pair(GLFW_KEY_KP_6, KeyboardKeyType::KP_6),
        std::make_pair(GLFW_KEY_KP_7, KeyboardKeyType::KP_7),
        std::make_pair(GLFW_KEY_KP_8, KeyboardKeyType::KP_8),
        std::make_pair(GLFW_KEY_KP_9, KeyboardKeyType::KP_9),
        std::make_pair(GLFW_KEY_KP_DECIMAL, KeyboardKeyType::KP_DECIMAL),
        std::make_pair(GLFW_KEY_KP_DIVIDE, KeyboardKeyType::KP_DIVIDE),
        std::make_pair(GLFW_KEY_KP_MULTIPLY, KeyboardKeyType::KP_MULTIPLY),
        std::make_pair(GLFW_KEY_KP_SUBTRACT, KeyboardKeyType::KP_SUBTRACT),
        std::make_pair(GLFW_KEY_KP_ADD, KeyboardKeyType::KP_ADD),
        std::make_pair(GLFW_KEY_KP_ENTER, KeyboardKeyType::KP_ENTER),
        std::make_pair(GLFW_KEY_KP_EQUAL, KeyboardKeyType::EQUAL),
        std::make_pair(GLFW_KEY_LEFT_SHIFT, KeyboardKeyType::LEFT_SHIFT),
        std::make_pair(GLFW_KEY_LEFT_CONTROL, KeyboardKeyType::LEFT_CONTROL),
        std::make_pair(GLFW_KEY_LEFT_ALT, KeyboardKeyType::LEFT_ALT),
        std::make_pair(GLFW_KEY_LEFT_SUPER, KeyboardKeyType::LEFT_SUPER),
        std::make_pair(GLFW_KEY_RIGHT_SHIFT, KeyboardKeyType::RIGHT_SHIFT),
        std::make_pair(GLFW_KEY_RIGHT_CONTROL, KeyboardKeyType::RIGHT_CONTROL),
        std::make_pair(GLFW_KEY_RIGHT_ALT, KeyboardKeyType::RIGHT_ALT),
        std::make_pair(GLFW_KEY_RIGHT_SUPER, KeyboardKeyType::RIGHT_SUPER),
        std::make_pair(GLFW_KEY_MENU, KeyboardKeyType::MENU)
    };
    return keyMap[key];
}

void GLFWCallbackDelegator::ErrorCallbackGLFW(int error, const char* description)
{
    METU_ERROR_LOG("GLFW %d: %s", error, description);
}

void GLFWCallbackDelegator::WindowPosGLFW(GLFWwindow* w, int x, int y)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend() && loc->second->input)
        loc->second->input->WindowPosChanged(x, y);
}

void GLFWCallbackDelegator::WindowFBGLFW(GLFWwindow* w, int width, int height)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend() && loc->second->input)
    {
        loc->second->viewportSize = Vector2i(width, height);
        loc->second->input->WindowFBChanged(width, height);
    }        
}

void GLFWCallbackDelegator::WindowSizeGLFW(GLFWwindow* w, int width, int height)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend() && loc->second->input)
    {        
        loc->second->vOpts.wSize = Vector2i(width, height);;
        loc->second->input->WindowSizeChanged(width, height);
    }
}

void GLFWCallbackDelegator::WindowCloseGLFW(GLFWwindow* w)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend() && loc->second->input)
    {
        loc->second->input->WindowClosed();
        loc->second->open = false;
    }
}

void GLFWCallbackDelegator::WindowRefreshGLFW(GLFWwindow* w)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend())
        loc->second->input->WindowRefreshed();
}

void GLFWCallbackDelegator::WindowFocusedGLFW(GLFWwindow* w, int b)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend() && loc->second->input)
        loc->second->input->WindowFocused(b);
}

void GLFWCallbackDelegator::WindowMinimizedGLFW(GLFWwindow* w, int b)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend() && loc->second->input)
        loc->second->input->WindowMinimized(b);
}

void GLFWCallbackDelegator::KeyboardUsedGLFW(GLFWwindow* w, int key, int scanCode,
                               int action, int modifiers)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend() && loc->second->input)
        loc->second->input->KeyboardUsed(DetermineKey(key),
                                         DetermineAction(action));
}

void GLFWCallbackDelegator::MouseMovedGLFW(GLFWwindow* w, double x, double y)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend() && loc->second->input)
        loc->second->input->MouseMoved(x, y);
}

void GLFWCallbackDelegator::MousePressedGLFW(GLFWwindow* w, int button, int action, int modifier)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend() && loc->second->input)
        loc->second->input->MouseButtonUsed(DetermineMouseButton(button),
                                            DetermineAction(action));
}

void GLFWCallbackDelegator::MouseScrolledGLFW(GLFWwindow* w, double x, double y)
{
    const auto& inputMap = Instance().windowMappings;
    auto loc = inputMap.find(w);
    if(loc != inputMap.cend() && loc->second->input)
        loc->second->input->MouseScrolled(x, y);
}

GLFWCallbackDelegator::GLFWCallbackDelegator()
{
    if(!glfwInit())
    {
        METU_ERROR_LOG("Could not Init GLFW");
        throw VisorError::WINDOW_GENERATOR_ERROR;
    }

    glfwSetErrorCallback(GLFWCallbackDelegator::ErrorCallbackGLFW);
}

GLFWCallbackDelegator::~GLFWCallbackDelegator()
{
    glfwTerminate();
}

void GLFWCallbackDelegator::AttachWindow(GLFWwindow* glfwWindow, VisorGL* window)
{
    auto ret = windowMappings.emplace(glfwWindow, window);
    // Override old callbacks
    if(!ret.second)
        ret.first->second = window;

    // Set Callbacks
    glfwSetWindowPosCallback(glfwWindow, GLFWCallbackDelegator::WindowPosGLFW);
    glfwSetFramebufferSizeCallback(glfwWindow, GLFWCallbackDelegator::WindowFBGLFW);
    glfwSetWindowSizeCallback(glfwWindow, GLFWCallbackDelegator::WindowSizeGLFW);
    glfwSetWindowCloseCallback(glfwWindow, GLFWCallbackDelegator::WindowCloseGLFW);
    glfwSetWindowRefreshCallback(glfwWindow, GLFWCallbackDelegator::WindowRefreshGLFW);
    glfwSetWindowFocusCallback(glfwWindow, GLFWCallbackDelegator::WindowFocusedGLFW);
    glfwSetWindowIconifyCallback(glfwWindow, GLFWCallbackDelegator::WindowMinimizedGLFW);

    glfwSetKeyCallback(glfwWindow, GLFWCallbackDelegator::KeyboardUsedGLFW);
    glfwSetCursorPosCallback(glfwWindow, GLFWCallbackDelegator::MouseMovedGLFW);
    glfwSetMouseButtonCallback(glfwWindow, GLFWCallbackDelegator::MousePressedGLFW);
    glfwSetScrollCallback(glfwWindow, GLFWCallbackDelegator::MouseScrolledGLFW);
}

void GLFWCallbackDelegator::DetachWindow(GLFWwindow* glfwWindow)
{
    windowMappings.erase(glfwWindow);
}