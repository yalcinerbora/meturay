#pragma once

#include <string>
#include "TracerOptionsI.h"

// Visor Related
enum class KeyboardKeyType;
enum class MouseButtonType;
enum class KeyAction;
enum class VisorActionType;
enum class TracerCameraMode;

// Tracer Related
enum class ScenePartitionerType;

// General
enum class PixelFormat;

namespace EnumStringConverter
{
    //================//
    // ENUM -> STRING //
    //================//
    // Visor Related
    std::string             KeyboardKeyTypeToString(KeyboardKeyType);
    std::string             MouseButtonTypeToString(MouseButtonType);
    std::string             KeyActionToString(KeyAction);
    std::string             VisorActionTypeToString(VisorActionType);
    std::string             CameraModeToString(TracerCameraMode);

    // Tracer Related
    std::string             ScenePartitionerTypeToString(ScenePartitionerType);

    // Tracer Option Related
    std::string             TracerOptionTypeToString(TracerOptionsI::OptionType);

    // General
    std::string             PixelFormatTypeToString(PixelFormat);

    //================//
    // STRING -> ENUM //
    //================//
    // Visor Related
    KeyboardKeyType             StringToKeyboardKeyType(const std::string&);
    MouseButtonType             StringToMouseButtonType(const std::string&);
    KeyAction                   StringToKeyAction(const std::string&);
    VisorActionType             StringToVisorActionType(const std::string&);
    TracerCameraMode            StringToCameraMode(const std::string&);

    // Tracer Related
    ScenePartitionerType        StringToScenePartitionerType(const std::string&);

    // Tracer Option Related
    TracerOptionsI::OptionType  StringToTracerOptionType(const std::string&);

    // General
    PixelFormat                 StringToPixelFormatType(const std::string&);
}