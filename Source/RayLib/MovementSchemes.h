#pragma once

#include "MovementSchemeI.h"

class MovementSchemeFPS final : public MovementSchemeI
{
    private:
        double              prevMouseX;
        double              prevMouseY;

        bool                mouseToggle;

        // Camera Movement Constants
        const double        Sensitivity;
        const double        MoveRatio;
        const double        MoveRatioModifier;

        double              currentMovementRatio;

    protected:
    public:
        // Constructors & Destructor
                            MovementSchemeFPS(double sensitivity,
                                              double moveRatio,
                                              double moveRatioModifier);

        // Interface
        bool                InputAction(VisorCamera&,
                                           VisorActionType,
                                           KeyAction) override;
        bool                MouseMovementAction(VisorCamera&,
                                                double x, double y) override;
        bool                MouseScrollAction(VisorCamera&,
                                              double x, double y) override;
};
