#pragma once

#include <map>

enum class KeyboardKeyType
{
    SPACE,
    APOSTROPHE,
    COMMA,
    MINUS,
    PERIOD,
    SLASH,
    NUMBER_0,
    NUMBER_1,
    NUMBER_2,
    NUMBER_3,
    NUMBER_4,
    NUMBER_5,
    NUMBER_6,
    NUMBER_7,
    NUMBER_8,
    NUMBER_9,
    SEMICOLON,
    EQUAL,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    LEFT_BRACKET,
    BACKSLASH,
    RIGHT_BRACKET,
    GRAVE_ACCENT,
    WORLD_1,
    WORLD_2,
    ESCAPE,
    ENTER,
    TAB,
    BACKSPACE,
    INSERT,
    DELETE_KEY,
    RIGHT,
    LEFT,
    DOWN,
    UP,
    PAGE_UP,
    PAGE_DOWN,
    HOME,
    END_KEY,
    CAPS_LOCK,
    SCROLL_LOCK,
    NUM_LOCK,
    PRINT_SCREEN,
    PAUSE,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
    F18,
    F19,
    F20,
    F21,
    F22,
    F23,
    F24,
    F25,
    KP_0,
    KP_1,
    KP_2,
    KP_3,
    KP_4,
    KP_5,
    KP_6,
    KP_7,
    KP_8,
    KP_9,
    KP_DECIMAL,
    KP_DIVIDE,
    KP_MULTIPLY,
    KP_SUBTRACT,
    KP_ADD,
    KP_ENTER,
    KP_EQUAL,
    LEFT_SHIFT,
    LEFT_CONTROL,
    LEFT_ALT,
    LEFT_SUPER,
    RIGHT_SHIFT,
    RIGHT_CONTROL,
    RIGHT_ALT,
    RIGHT_SUPER,
    MENU,

    END
};

enum class MouseButtonType
{
    LEFT,
    RIGHT,
    MIDDLE,
    BUTTON_4,
    BUTTON_5,
    BUTTON_6,
    BUTTON_7,
    BUTTON_8,

    END
};

enum class KeyAction
{
    PRESSED,
    RELEASED,
    REPEATED,

    END
};

enum class VisorActionType
{
    // Movement Related
    MOVE_FORWARD,
    MOVE_BACKWARD,
    MOVE_RIGHT,
    MOVE_LEFT,
    MOUSE_MOVE_MODIFIER,
    MOUSE_TRANSLATE_MODIFIER,
    FAST_MOVE_MODIFIER,
    // Change Camera Movement Type
    MOVE_TYPE_NEXT,
    MOVE_TYPE_PREV,
    // Camera Related
    // Enable Disable Camera Movement
    TOGGLE_CUSTOM_SCENE_CAMERA,
    LOCK_UNLOCK_CAMERA,
    PRINT_CUSTOM_CAMERA,
    //
    SCENE_CAM_NEXT,
    SCENE_CAM_PREV,
    // Start Stop Actions
    START_STOP_TRACE,
    PAUSE_CONT_TRACE,
    // Animation Related
    FRAME_NEXT,
    FRAME_PREV,
    // Image Related
    SAVE_IMAGE,
    SAVE_IMAGE_HDR,
    // Lifetime Related
    CLOSE,

    END
};

using KeyboardKeyBindings = std::multimap<KeyboardKeyType, VisorActionType>;
using MouseKeyBindings = std::multimap<MouseButtonType, VisorActionType>;