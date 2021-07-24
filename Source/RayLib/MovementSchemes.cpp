#include "MovementSchemes.h"
#include "Vector.h"
#include "Quaternion.h"
#include "VisorCamera.h"

MovementSchemeFPS::MovementSchemeFPS(double sensitivity,
                                     double moveRatio,
                                     double moveRatioModifier)
    : Sensitivity(sensitivity)
    , MoveRatio(moveRatio)
    , MoveRatioModifier(moveRatioModifier)
    , prevMouseX(0.0)
    , prevMouseY(0.0)
    , mouseToggle(false)
    , currentMovementRatio(MoveRatio)
{}

// Interface
bool MovementSchemeFPS::InputAction(VisorCamera& camera,
                                    VisorActionType visorAction,
                                    KeyAction action)
{
    // Shift modifier
    if(action == KeyAction::PRESSED && visorAction == VisorActionType::FAST_MOVE_MODIFIER)
    {
        currentMovementRatio = MoveRatio * MoveRatioModifier;
        return false;
    }
    else if(action == KeyAction::RELEASED && visorAction == VisorActionType::FAST_MOVE_MODIFIER)
    {
        currentMovementRatio = MoveRatio;
        return false;
    }

    if(visorAction == VisorActionType::MOUSE_MOVE_MODIFIER)
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
                camera.position += lookDir * static_cast<float>(currentMovementRatio);
                camera.gazePoint += lookDir * static_cast<float>(currentMovementRatio);
                break;
            }
            case VisorActionType::MOVE_LEFT:
            {
                camera.position += side * static_cast<float>(currentMovementRatio);
                camera.gazePoint += side * static_cast<float>(currentMovementRatio);
                break;
            }
            case VisorActionType::MOVE_BACKWARD:
            {
                camera.position += lookDir * static_cast<float>(-currentMovementRatio);
                camera.gazePoint += lookDir * static_cast<float>(-currentMovementRatio);
                break;
            }
            case VisorActionType::MOVE_RIGHT:
            {
                camera.position += side * static_cast<float>(-currentMovementRatio);
                camera.gazePoint += side * static_cast<float>(-currentMovementRatio);
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

bool MovementSchemeFPS::MouseMovementAction(VisorCamera& camera,
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

bool MovementSchemeFPS::MouseScrollAction(VisorCamera&,
                                          double x, double y)
{
    return false;
}

        // Constructors & Destructor
MovementSchemeMaya::MovementSchemeMaya(double sensitivity,
                                       double zoomPercentage,
                                       double translateModifier)
    : Sensitivity(sensitivity)
    , ZoomPercentage(zoomPercentage)
    , TranslateModifier(translateModifier)
    , moveMode(false)
    , translateMode(false)
    , mouseX(0.0)
    , mouseY(0.0)
{

}

// Interface
bool MovementSchemeMaya::InputAction(VisorCamera&,
                                     VisorActionType visorAction,
                                     KeyAction action)
{
    switch(visorAction)
    {
        case VisorActionType::MOUSE_MOVE_MODIFIER:
        {
            moveMode = (action == KeyAction::RELEASED) ? false : true;
            break;
        }
        case VisorActionType::MOUSE_TRANSLATE_MODIFIER:
        {
            translateMode = (action == KeyAction::RELEASED) ? false : true;
            break;
        }
        default: return false;
    }
    return false;
}

bool MovementSchemeMaya::MouseMovementAction(VisorCamera& camera,
                                             double x, double y)
{
    bool camChanged = false;
    // Check with latest recorded input
	float diffX = static_cast<float>(x - mouseX);
    float diffY = static_cast<float>(y - mouseY);

	if(moveMode)
	{
		// X Rotation
		Vector3f lookDir = camera.gazePoint - camera.position;
		QuatF rotateX(static_cast<float>(-diffX * Sensitivity), YAxis);
        Vector3f rotated = rotateX.ApplyRotation(lookDir);
		camera.position = camera.gazePoint - rotated;

		// Y Rotation
		lookDir = camera.gazePoint - camera.position;
        Vector3f left = Cross(camera.up, lookDir).NormalizeSelf();
        QuatF rotateY(static_cast<float>(diffY * Sensitivity), left);
		rotated = rotateY.ApplyRotation((lookDir));
		camera.position = camera.gazePoint - rotated;

		// Redefine up
		// Enforce an up vector which is ortogonal to the xz plane
		camera.up = Cross(rotated, left);
		camera.up[2] = 0.0f;
		camera.up[0] = 0.0f;
		camera.up.NormalizeSelf();
        camChanged = true;
	}
	if(translateMode)
	{
        Vector3f lookDir = camera.gazePoint - camera.position;
        Vector3f side = Cross(camera.up, lookDir).NormalizeSelf();
		camera.position += static_cast<float>(diffX * TranslateModifier) * side;
		camera.gazePoint += static_cast<float>(diffX * TranslateModifier) * side;

		camera.position += static_cast<float>(diffY * TranslateModifier) * camera.up;
		camera.gazePoint += static_cast<float>(diffY * TranslateModifier) * camera.up;
        camChanged = true;
	}

	mouseX = x;
	mouseY = y;
    return camChanged;
}

bool MovementSchemeMaya::MouseScrollAction(VisorCamera& camera,
                                           double x, double y)
{
    // Zoom to the focus until some threshold
    Vector3f lookDir = camera.position - camera.gazePoint;
    lookDir *= static_cast<float>(1.0 - y * ZoomPercentage);
    if(lookDir.Length() > 0.1f)
    {
        camera.position = lookDir + camera.gazePoint;
        return true;
    }
    return false;
}