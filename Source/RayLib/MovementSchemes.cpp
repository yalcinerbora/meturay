#include "MovementSchemes.h"
#include "Vector.h"
#include "Quaternion.h"
#include "Camera.h"

MovementSchemeFPS::MovementSchemeFPS(double sensitivity,
                                     double moveRatio,
                                     double moveRatioModifier)
    : Sensitivity(sensitivity)
    , MoveRatio(moveRatio)
    , MoveRatioModifier(moveRatioModifier)
    , prevMouseX(0.0)
    , prevMouseY(0.0)
    , mouseToggle(false)
{}

// Interface
bool MovementSchemeFPS::InputAction(CameraPerspective& camera,
                                    VisorActionType visorAction,
                                    KeyAction action)
{
    // Shift modifier
    double currentRatio = 0.0;
    if(action == KeyAction::PRESSED && visorAction == VisorActionType::FAST_MOVE_TOGGLE)
    {
        currentMovementRatio = MoveRatio * MoveRatioModifier;
        return false;
    }
    else if(action == KeyAction::RELEASED && visorAction == VisorActionType::FAST_MOVE_TOGGLE)
    {
        currentMovementRatio = MoveRatio;
        return false;
    }

    if(visorAction == VisorActionType::MOUSE_MOVE_TOGGLE)
    {
        mouseToggle = (action == KeyAction::RELEASED) ? false : true;
        return false;
    }

    if(action != KeyAction::RELEASED)
    {
        bool camChanged = true;
        Vector3 lookDir = (camera.gazePoint - camera.position).NormalizeSelf();
        Vector3 side = Cross(camera.up, lookDir).NormalizeSelf();
        switch(visorAction)
        {
            // Movement
            case VisorActionType::MOVE_FORWARD:
            {
                camera.position += lookDir * static_cast<float>(currentRatio);
                camera.gazePoint += lookDir * static_cast<float>(currentRatio);
                break;
            }
            case VisorActionType::MOVE_LEFT:
            {
                camera.position += side * static_cast<float>(currentRatio);
                camera.gazePoint += side * static_cast<float>(currentRatio);
                break;
            }
            case VisorActionType::MOVE_BACKWARD:
            {
                camera.position += lookDir * static_cast<float>(-currentRatio);
                camera.gazePoint += lookDir * static_cast<float>(-currentRatio);
                break;
            }
            case VisorActionType::MOVE_RIGHT:
            {
                camera.position += side * static_cast<float>(-currentRatio);
                camera.gazePoint += side * static_cast<float>(-currentRatio);
                break;
            }
            default:
                // Do nothing on other keys
                camChanged = false;
                break;
        }
        return camChanged;
    }
    return false;
}

bool MovementSchemeFPS::MouseMovementAction(CameraPerspective& camera,
                                            double x, double y)
{
    // Check with latest recorded input
    double diffX = x - prevMouseX;
    double diffY = y - prevMouseY;

    if(mouseToggle)
    {
        // X Rotation
        Vector3 lookDir = camera.gazePoint - camera.position;
        QuatF rotateX(static_cast<float>(-diffX * Sensitivity), YAxis);
        Vector3 rotated = rotateX.ApplyRotation(lookDir);
        camera.gazePoint = camera.position + rotated;

        // Y Rotation
        lookDir = camera.gazePoint - camera.position;
        Vector3 side = Cross(camera.up, lookDir).NormalizeSelf();
        QuatF rotateY(static_cast<float>(diffY * Sensitivity), side);
        rotated = rotateY.ApplyRotation((lookDir));
        camera.gazePoint = camera.position + rotated;

        // Redefine up
        // Enforce an up vector which is ortogonal to the xz plane
        camera.up = Cross(rotated, side);
        camera.up[0] = 0.0f;
        camera.up[1] = (camera.up[1] < 0.0f) ? -1.0f : 1.0f;
        camera.up[2] = 0.0f;
    }
    prevMouseX = x;
    prevMouseY = y;
    return mouseToggle;
}

bool MovementSchemeFPS::MouseScrollAction(CameraPerspective&,
                                          double x, double y)
{
    return false;
}