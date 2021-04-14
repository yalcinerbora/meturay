#pragma once

#include "MovementSchemeI.h"

class MovementSchemeFPS final : public MovementSchemeI
{
    public:
        static constexpr double     DefaultSensitivity          = 1.0;
        static constexpr double     DefaultMoveRatio            = 1.0;
        static constexpr double     DefaultMoveRatioModifier    = 2.5;
    private:
        double                      prevMouseX;
        double                      prevMouseY;

        bool                        mouseToggle;

        // Camera Movement Constants
        const double                Sensitivity;
        const double                MoveRatio;
        const double                MoveRatioModifier;

        double                      currentMovementRatio;

    protected:
    public:
        // Constructors & Destructor
                            MovementSchemeFPS(double sensitivity = DefaultSensitivity,
                                              double moveRatio = DefaultMoveRatio,
                                              double moveRatioModifier = DefaultMoveRatioModifier);

        // Interface
        bool                InputAction(VisorCamera&,
                                           VisorActionType,
                                           KeyAction) override;
        bool                MouseMovementAction(VisorCamera&,
                                                double x, double y) override;
        bool                MouseScrollAction(VisorCamera&,
                                              double x, double y) override;
};
