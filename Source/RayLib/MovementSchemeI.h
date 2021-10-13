#pragma once

#include "VisorInputStructs.h"

struct VisorTransform;

class MovementSchemeI
{
    public:
        virtual                 ~MovementSchemeI() = default;

        // Interface
        virtual bool            InputAction(VisorTransform&,
                                            VisorActionType,
                                            KeyAction) = 0;

        virtual bool            MouseMovementAction(VisorTransform&,
                                                    double x, double y) = 0;
        virtual bool            MouseScrollAction(VisorTransform&,
                                                  double x, double y) = 0;
};