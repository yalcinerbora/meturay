#pragma once

#include "MovementSchemeI.h"

class MovementSchemeFPS final : public MovementSchemeI
{
    public:
        static constexpr double     DefaultSensitivity          = 0.0025;
        static constexpr double     DefaultMoveRatio            = 1.5;
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
        bool                InputAction(VisorTransform&,
                                        VisorActionType,
                                        KeyAction) override;
        bool                MouseMovementAction(VisorTransform&,
                                                double x, double y) override;
        bool                MouseScrollAction(VisorTransform&,
                                              double x, double y) override;
};

class MovementSchemeMaya final : public MovementSchemeI
{
    public:
        static constexpr double     DefaultSensitivity          = 0.0025;
        static constexpr double     DefaultZoomPercentage       = 0.1;
        static constexpr double     DefaultTranslateModifier    = 0.01;
    private:
        // Camera Movement Constants
        const double            Sensitivity;
        const double            ZoomPercentage;
        const double            TranslateModifier;

        bool						moveMode;
        bool						translateMode;
        double					mouseX;
        double					mouseY;

    protected:
    public:
        // Constructors & Destructor
                            MovementSchemeMaya(double sensitivity = DefaultSensitivity,
                                              double zoomPercentage = DefaultZoomPercentage,
                                              double translateModifier = DefaultTranslateModifier);

        // Interface
        bool                InputAction(VisorTransform&,
                                        VisorActionType,
                                        KeyAction) override;
        bool                MouseMovementAction(VisorTransform&,
                                                double x, double y) override;
        bool                MouseScrollAction(VisorTransform&,
                                              double x, double y) override;
};