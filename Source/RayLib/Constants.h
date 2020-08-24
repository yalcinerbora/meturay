#pragma once
/**

Many Constants that are used throught the tracer

*/

#include <limits>
#include <array>

#include "Vector.h"
#include "TracerStructs.h"
#include "VisorInputStructs.h"


namespace ProgramConstants
{
    static constexpr std::array<std::array<const char* const, 3>, 1> ContributorList =
    {
        std::array<const char* const, 3>{"Bora Yalciner", "METU, Turkey", "yalciner.bora@metu.edu.tr"},
    };

    static const std::string ContributorCompaq = "Yalciner B.";

    static const std::string ProgramName = "METURay";
    static const std::string VersionMajor = "0";
    static const std::string VersionMinor = "1";

    static const std::string Version = ("v" + VersionMajor + "."
                                        + ((VersionMinor.size() == 1) ? "0" : "")
                                        + VersionMinor);

    static const std::string LicenseName = "MIT";
    static const std::string Footer = ("This executable is provided as a part of " + ProgramName
                                       + " " + Version + " which is developed by "
                                       + ContributorCompaq + " and offered with " + LicenseName
                                       + " License. Please check *_LICENSE files for used third-party "
                                       + "licenses");



}

namespace VisorConstants
{
    static const KeyboardKeyBindings DefaultKeyBinds =
    {
        {KeyboardKeyType::W, VisorActionType::MOVE_FORWARD},
        {KeyboardKeyType::S, VisorActionType::MOVE_BACKWARD},
        {KeyboardKeyType::D, VisorActionType::MOVE_RIGHT},
        {KeyboardKeyType::A, VisorActionType::MOVE_LEFT},

        {KeyboardKeyType::LEFT_SHIFT, VisorActionType::FAST_MOVE_TOGGLE},

        {KeyboardKeyType::KP_9, VisorActionType::MOVE_TYPE_NEXT},
        {KeyboardKeyType::KP_7, VisorActionType::MOVE_TYPE_PREV},

        {KeyboardKeyType::KP_5, VisorActionType::TOGGLE_CUSTOM_SCENE_CAMERA},
        {KeyboardKeyType::L, VisorActionType::LOCK_UNLOCK_CAMERA},
        //
        {KeyboardKeyType::KP_6, VisorActionType::SCENE_CAM_NEXT},
        {KeyboardKeyType::KP_4, VisorActionType::SCENE_CAM_PREV},
        // Start Stop Actions
        {KeyboardKeyType::O, VisorActionType::START_STOP_TRACE},
        {KeyboardKeyType::P, VisorActionType::PAUSE_CONT_TRACE},
        // Animation Related
        {KeyboardKeyType::RIGHT, VisorActionType::FRAME_NEXT},
        {KeyboardKeyType::LEFT, VisorActionType::FRAME_PREV},
        // Image Related
        {KeyboardKeyType::SPACE, VisorActionType::SAVE_IMAGE},
        // Lifetime Realted
        {KeyboardKeyType::ESCAPE, VisorActionType::CLOSE}
    };

    static const MouseKeyBindings DefaultButtonBinds =
    {
        {MouseButtonType::LEFT, VisorActionType::MOUSE_MOVE_TOGGLE},
        {MouseButtonType::RIGHT, VisorActionType::MOUSE_MOVE_TOGGLE},

        {MouseButtonType::MIDDLE, VisorActionType::LOCK_UNLOCK_CAMERA}
    };
}

namespace TracerConstants
{
    //static constexpr TracerCommonOpts DefaultTracerCommonOptions = {false};
}

namespace BaseConstants
{
    static constexpr const char* EMPTY_PRIMITIVE_NAME = "Empty";
    static constexpr Vector2i IMAGE_MAX_SIZE = Vector2i(std::numeric_limits<int>::max(),
                                                        std::numeric_limits<int>::max());
}

namespace SceneConstants
{
    // Fundamental Limitations (for convenience)
    static constexpr const int  MaxPrimitivePerSurface = 8;
}

namespace MathConstants
{
    static constexpr double Pi_d = 3.1415926535897932384626433;
    static constexpr double PiSqr_d = Pi_d * Pi_d;
    static constexpr double InvPi_d = 1.0 / Pi_d;
    static constexpr double InvPiSqr_d = 1.0 / (Pi_d * Pi_d);
    static constexpr double Sqrt2_d = 1.4142135623730950488016887;
    static constexpr double Sqrt3_d = 1.7320508075688772935274463;
    static constexpr double E_d = 2.7182818284590452353602874;
    static constexpr double InvE_d = 1.0 / E_d;

    static constexpr double DegToRadCoef_d = Pi_d / 180.0;
    static constexpr double RadToDegCoef_d = 180.0 / Pi_d;

    static constexpr double Epsilon_d = 1.0e-6;

    static constexpr float Pi = static_cast<float>(Pi_d);
    static constexpr float PiSqr = static_cast<float>(PiSqr_d);
    static constexpr float InvPi = static_cast<float>(InvPi_d);
    static constexpr float InvPiSqr = static_cast<float>(InvPiSqr_d);
    static constexpr float Sqrt2 = static_cast<float>(Sqrt2_d);
    static constexpr float Sqrt3 = static_cast<float>(Sqrt3_d);
    static constexpr float E = static_cast<float>(E_d);
    static constexpr float InvE = static_cast<float>(InvE_d);

    static constexpr float DegToRadCoef = static_cast<float>(DegToRadCoef_d);
    static constexpr float RadToDegCoef = static_cast<float>(RadToDegCoef_d);

    static constexpr float Epsilon = static_cast<float>(Epsilon_d);
}
