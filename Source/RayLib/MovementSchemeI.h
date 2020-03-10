#pragma once

#include "VisorInputStructs.h"

struct CPUCamera;

class MovementSchemeI
{
    public:
        virtual                 ~MovementSchemeI() = default;

        // Interface
        virtual bool            InputAction(CPUCamera&,
                                            VisorActionType,
                                            KeyAction) = 0;

        virtual bool            MouseMovementAction(CPUCamera&,
                                                    double x, double y) = 0;
        virtual bool            MouseScrollAction(CPUCamera&,
                                                  double x, double y) = 0;
};