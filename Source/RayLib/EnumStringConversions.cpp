#include "EnumStringConversions.h"

#include "TracerSystemI.h"
#include "VisorI.h"

template<class T>
using EnumNameList = std::array<const char* const, static_cast<size_t>(T::END)>;

static constexpr EnumNameList<ScenePartitionerType> ScenePartitionerTypeNames =
{
    "SINGLE_GPU",
    "MULTI_GPU"
};

static constexpr EnumNameList<OptionsI::OptionType> TracerOptionTypeNames =
{
    "bool",
    "int",
    "float",
    "vec2l",
    "vec2",
    "vec3",
    "vec4",
    "string"
};

static constexpr EnumNameList<OutputMetric> OutputMetricNames =
{
    "Time",
    "Sample",
    "Both"
};

static constexpr EnumNameList<TracerCameraMode> CameraModeTypeNames =
{
    "SCENE_CAM",
    "CUSTOM_CAM"
};

static constexpr EnumNameList<KeyAction> KeyActionTypeNames =
{
    "PRESSED"
    "RELEASED"
    "REPEATED"
};

static constexpr EnumNameList<MouseButtonType> MouseButtonTypeNames =
{
    "LEFT",
    "RIGHT",
    "MIDDLE",
    "BUTTON_4",
    "BUTTON_5",
    "BUTTON_6",
    "BUTTON_7",
    "BUTTON_8"
};

static constexpr EnumNameList<KeyboardKeyType> KeyboardKeyTypeNames =
{
    "SPACE",
    "APOSTROPHE",
    "COMMA",
    "MINUS",
    "PERIOD",
    "SLASH",
    "NUMBER_0",
    "NUMBER_1",
    "NUMBER_2",
    "NUMBER_3",
    "NUMBER_4",
    "NUMBER_5",
    "NUMBER_6",
    "NUMBER_7",
    "NUMBER_8",
    "NUMBER_9",
    "SEMICOLON",
    "EQUAL",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "LEFT_BRACKET",
    "BACKSLASH",
    "RIGHT_BRACKET",
    "GRAVE_ACCENT",
    "WORLD_1",
    "WORLD_2",
    "ESCAPE",
    "ENTER",
    "TAB",
    "BACKSPACE",
    "INSERT",
    "DELETE_KEY",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UP",
    "PAGE_UP",
    "PAGE_DOWN",
    "HOME",
    "END_KEY",
    "CAPS_LOCK",
    "SCROLL_LOCK",
    "NUM_LOCK",
    "PRINT_SCREEN",
    "PAUSE",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "F13",
    "F14",
    "F15",
    "F16",
    "F17",
    "F18",
    "F19",
    "F20",
    "F21",
    "F22",
    "F23",
    "F24",
    "F25",
    "KP_0",
    "KP_1",
    "KP_2",
    "KP_3",
    "KP_4",
    "KP_5",
    "KP_6",
    "KP_7",
    "KP_8",
    "KP_9",
    "KP_DECIMAL",
    "KP_DIVIDE",
    "KP_MULTIPLY",
    "KP_SUBTRACT",
    "KP_ADD",
    "KP_ENTER",
    "KP_EQUAL",
    "LEFT_SHIFT",
    "LEFT_CONTROL",
    "LEFT_ALT",
    "LEFT_SUPER",
    "RIGHT_SHIFT",
    "RIGHT_CONTROL",
    "RIGHT_ALT",
    "RIGHT_SUPER",
    "MENU"
};

static constexpr EnumNameList<VisorActionType> VisorActionTypeNames =
{
    // Movement Related
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "MOUSE_MOVE_TOGGLE",
    "FAST_MOVE_TOGGLE",
    // Change Camera Movement Type
    "MOVE_TYPE_NEXT",
    "MOVE_TYPE_PREV",
    // Camera Related
    // Enable Disable Camera Movement
    "TOGGLE_CUSTOM_SCENE_CAMERA",
    "LOCK_UNLOCK_CAMERA",
    //
    "SCENE_CAM_NEXT",
    "SCENE_CAM_PREV",
    // Start Stop Actions
    "START_STOP_TRACE",
    "PAUSE_CONT_TRACE",
    // Animation Related
    "FRAME_NEXT",
    "FRAME_PREV",
    // Image Related
    "SAVE_IMAGE",
    // Lifetime Related
    "CLOSE"
};

static constexpr EnumNameList<PixelFormat> PixelFormatTypeNames =
{
    "R8_UNORM",
    "RG8_UNORM",
    "RGB8_UNORM",
    "RGBA8_UNORM",

    "R16_UNORM",
    "RG16_UNORM",
    "RGB16_UNORM",
    "RGBA16_UNORM",

    "R_HALF",
    "RG_HALF",
    "RGB_HALF",
    "RGBA_HALF",

    "R_FLOAT",
    "RG_FLOAT",
    "RGB_FLOAT",
    "RGBA_FLOAT"
};

template<class T>
inline T StrToEnum(const std::string& s,
                   const EnumNameList<T>& nameList)
{
    for(uint32_t i = 0; i < nameList.size(); i++)
    {
        std::string str = nameList[i];
        if(str == s) return static_cast<T>(i);
    }
    return T::END;
}

std::string EnumStringConverter::KeyboardKeyTypeToString(KeyboardKeyType t)
{
    return KeyboardKeyTypeNames[static_cast<uint32_t>(t)];
}

std::string EnumStringConverter::MouseButtonTypeToString(MouseButtonType t)
{
    return TracerOptionTypeNames[static_cast<uint32_t>(t)];
}

std::string EnumStringConverter::KeyActionToString(KeyAction t)
{
    return KeyActionTypeNames[static_cast<uint32_t>(t)];
}

std::string EnumStringConverter::VisorActionTypeToString(VisorActionType t)
{
    return VisorActionTypeNames[static_cast<uint32_t>(t)];
}

std::string EnumStringConverter::CameraModeToString(TracerCameraMode t)
{
    return CameraModeTypeNames[static_cast<uint32_t>(t)];
}

std::string EnumStringConverter::OutputMetricToString(OutputMetric t)
{
    return OutputMetricNames[static_cast<uint32_t>(t)];
}

std::string EnumStringConverter::PixelFormatTypeToString(PixelFormat t)
{
    return PixelFormatTypeNames[static_cast<uint32_t>(t)];
}

std::string EnumStringConverter::ScenePartitionerTypeToString(ScenePartitionerType t)
{
    return ScenePartitionerTypeNames[static_cast<uint32_t>(t)];
}

std::string EnumStringConverter::OptionTypeToString(OptionsI::OptionType t)
{
    return TracerOptionTypeNames[static_cast<uint32_t>(t)];
}

KeyboardKeyType EnumStringConverter::StringToKeyboardKeyType(const std::string& s)
{
    return StrToEnum<KeyboardKeyType>(s, KeyboardKeyTypeNames);
}

MouseButtonType EnumStringConverter::StringToMouseButtonType(const std::string& s)
{
    return StrToEnum<MouseButtonType>(s, MouseButtonTypeNames);
}

KeyAction EnumStringConverter::StringToKeyAction(const std::string& s)
{
    return StrToEnum<KeyAction>(s, KeyActionTypeNames);
}

VisorActionType EnumStringConverter::StringToVisorActionType(const std::string& s)
{
    return StrToEnum<VisorActionType>(s, VisorActionTypeNames);
}

TracerCameraMode EnumStringConverter::StringToCameraMode(const std::string& s)
{
    return StrToEnum<TracerCameraMode>(s, CameraModeTypeNames);
}

OutputMetric EnumStringConverter::StringToOutputMetric(const std::string& s)
{
    return StrToEnum<OutputMetric>(s, OutputMetricNames);
}

ScenePartitionerType EnumStringConverter::StringToScenePartitionerType(const std::string& s)
{
    return StrToEnum<ScenePartitionerType>(s, ScenePartitionerTypeNames);
}

OptionsI::OptionType EnumStringConverter::StringToOptionType(const std::string& s)
{
    return StrToEnum<OptionsI::OptionType>(s, TracerOptionTypeNames);
}

PixelFormat EnumStringConverter::StringToPixelFormatType(const std::string& s)
{
    return StrToEnum<PixelFormat>(s, PixelFormatTypeNames);
}