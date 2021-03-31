#pragma once
/**

MVisorInput Interface

Can be attached to a Visor to capture window actions

*/

#include <functional>
#include "VisorInputStructs.h"

class VisorCallbacksI;

using KeyCallbacks = std::multimap<std::pair<KeyboardKeyType, KeyAction>, std::function<void()>>;
using MouseButtonCallbacks = std::multimap<std::pair<MouseButtonType, KeyAction>, std::function<void()>>;

class VisorInputI
{
    private:
        KeyCallbacks                        keyCallbacks;
        MouseButtonCallbacks                buttonCallbacks;

        void                                KeyboardUsedWithCallbacks(KeyboardKeyType key, KeyAction action);
        void                                MouseButtonUsedWithCallbacks(MouseButtonType button, KeyAction action);

    protected:
    public:
        virtual                             ~VisorInputI() = default;

        // Interface
        virtual void                        AttachVisorCallback(VisorCallbacksI&) = 0;

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

template <class Function, class... Args>
void VisorInputI::AddKeyCallback(KeyboardKeyType key, KeyAction action,
                                 Function&& f, Args&&... args)
{
    std::function<void()> func = std::bind(f, args...);
    keyCallbacks.emplace(std::make_pair(key, action), func);
}

template <class Function, class... Args>
void VisorInputI::AddButtonCallback(MouseButtonType button, KeyAction action,
                                    Function&& f, Args&&... args)
{
    std::function<void()> func = std::bind(f, args...);
    buttonCallbacks.emplace(std::make_pair(button, action), func);
}

inline void VisorInputI::KeyboardUsedWithCallbacks(KeyboardKeyType key, KeyAction action)
{
    KeyboardUsed(key, action);

    auto range = keyCallbacks.equal_range(std::make_pair(key, action));
    for(auto it = range.first; it != range.second; ++it)
    {
        // Call Those Functions
        it->second();
    }
}

inline void VisorInputI::MouseButtonUsedWithCallbacks(MouseButtonType button, KeyAction action)
{
    MouseButtonUsed(button, action);

    auto range = buttonCallbacks.equal_range(std::make_pair(button, action));
    for(auto it = range.first; it != range.second; ++it)
    {
        // Call Those Functions
        it->second();
    }
}