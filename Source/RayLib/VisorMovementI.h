#pragma once

#include "VisorInputStructs.h"

struct CameraPerspective;

class MovementSchemeI
{
    public:
        virtual                 ~MovementSchemeI() = default;

        // Interface
        virtual void            KeyboardAction(CameraPerspective&,
                                               // ..
                                               KeyboardKeyType,
                                               KeyAction,
                                               const KeyboardKeyBindings&) = 0;

        virtual void            MouseMovementAction(CameraPerspective&,
                                                    // ..
                                                    double x, double y) = 0;
        virtual void            MouseScrollAction(CameraPerspective&,
                                                  // ..
                                                  double x, double y) = 0;
        virtual void            MouseButtonAction(CameraPerspective&,
                                                  // ..
                                                  MouseButtonType button,
                                                  KeyAction action,
                                                  const MouseKeyBindings&) = 0;
};