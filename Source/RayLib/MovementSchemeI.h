#pragma once

#include "VisorInputStructs.h"

struct CameraPerspective;

class MovementSchemeI
{
    public:
        virtual                 ~MovementSchemeI() = default;

        // Interface
        virtual bool            InputAction(CameraPerspective&,
                                            VisorActionType,
                                            KeyAction) = 0;

        virtual bool            MouseMovementAction(CameraPerspective&,
                                                    double x, double y) = 0;
        virtual bool            MouseScrollAction(CameraPerspective&,
                                                  double x, double y) = 0;
};