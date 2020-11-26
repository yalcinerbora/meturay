#pragma once

#include "VisorInputStructs.h"

struct VisorCamera;

class MovementSchemeI
{
    public:
        virtual                 ~MovementSchemeI() = default;

        // Interface
        virtual bool            InputAction(VisorCamera&,
                                            VisorActionType,
                                            KeyAction) = 0;

        virtual bool            MouseMovementAction(VisorCamera&,
                                                    double x, double y) = 0;
        virtual bool            MouseScrollAction(VisorCamera&,
                                                  double x, double y) = 0;
};