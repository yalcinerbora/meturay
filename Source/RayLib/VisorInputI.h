#pragma once
/**

MVisorInput Interface

Can be attached to a Visor to capture window actions

*/

#include <cstdint>
#include <functional>
#include "VisorInputStructs.h"

class VisorCallbacksI;
struct VisorTransform;
struct SceneAnalyticData;
struct TracerAnalyticData;
class TracerOptions;
struct TracerParameters;

using KeyCallbacks = std::multimap<std::pair<KeyboardKeyType, KeyAction>, std::function<void()>>;
using MouseButtonCallbacks = std::multimap<std::pair<MouseButtonType, KeyAction>, std::function<void()>>;

class VisorI;
struct TracerState;

class WindowInputI
{
    private:
    protected:
        KeyCallbacks                        keyCallbacks;
        MouseButtonCallbacks                buttonCallbacks;

        void                                KeyboardUsedWithCallbacks(KeyboardKeyType key, KeyAction action);
        void                                MouseButtonUsedWithCallbacks(MouseButtonType button, KeyAction action);

    public:
        virtual                             ~WindowInputI() = default;

        // Interface
        virtual void                        WindowPosChanged(int posX, int posY) = 0;
        virtual void                        WindowFBChanged(int fbWidth, int fbHeight) = 0;
        virtual void                        WindowSizeChanged(int width, int height) = 0;
        virtual void                        WindowClosed() = 0;
        virtual void                        WindowRefreshed() = 0;
        virtual void                        WindowFocused(bool) = 0;
        virtual void                        WindowMinimized(bool) = 0;

        virtual void                        MouseScrolled(double xOffset, double yOffset) = 0;
        virtual void                        MouseMoved(double x, double y) = 0;

        virtual void                        KeyboardUsed(KeyboardKeyType key, KeyAction action) = 0;
        virtual void                        MouseButtonUsed(MouseButtonType button, KeyAction action) = 0;

        // Defining Custom Callback
        template <class Function, class... Args>
        void                                AddKeyCallback(KeyboardKeyType, KeyAction,
                                                           Function&& f, Args&&... args);
        template <class Function, class... Args>
        void                                AddButtonCallback(MouseButtonType, KeyAction,
                                                              Function&& f, Args&&... args);
};

class VisorInputI : public WindowInputI
{
    public:
        virtual             ~VisorInputI() = default;
        // Interface
        // Renders GUI if available (GUI considered as an input interface)
        virtual void        RenderGUI() = 0;

        // Setters
        virtual void        SetTransform(const VisorTransform&) = 0;
        virtual void        SetSceneCameraCount(uint32_t) = 0;
        virtual void        SetSceneAnalyticData(const SceneAnalyticData&) = 0;
        virtual void        SetTracerAnalyticData(const TracerAnalyticData&) = 0;
        virtual void        SetTracerOptions(const TracerOptions&) = 0;
        virtual void        SetTracerParams(const TracerParameters&) = 0;
};

template <class Function, class... Args>
void WindowInputI::AddKeyCallback(KeyboardKeyType key, KeyAction action,
                                 Function&& f, Args&&... args)
{
    std::function<void()> func = std::bind(f, args...);
    keyCallbacks.emplace(std::make_pair(key, action), func);
}

template <class Function, class... Args>
void WindowInputI::AddButtonCallback(MouseButtonType button, KeyAction action,
                                    Function&& f, Args&&... args)
{
    std::function<void()> func = std::bind(f, args...);
    buttonCallbacks.emplace(std::make_pair(button, action), func);
}

inline void WindowInputI::KeyboardUsedWithCallbacks(KeyboardKeyType key, KeyAction action)
{
    KeyboardUsed(key, action);

    auto range = keyCallbacks.equal_range(std::make_pair(key, action));
    for(auto it = range.first; it != range.second; ++it)
    {
        // Call Those Functions
        it->second();
    }
}

inline void WindowInputI::MouseButtonUsedWithCallbacks(MouseButtonType button, KeyAction action)
{
    MouseButtonUsed(button, action);

    auto range = buttonCallbacks.equal_range(std::make_pair(button, action));
    for(auto it = range.first; it != range.second; ++it)
    {
        // Call Those Functions
        it->second();
    }
}