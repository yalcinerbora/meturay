#include "MovementSchemes.h"
#include "Vector.h"
#include "Quaternion.h"
#include "VisorTransform.h"

MovementSchemeFPS::MovementSchemeFPS(double sensitivity,
                                     double moveRatio,
                                     double moveRatioModifier)
    : prevMouseX(0.0)
    , prevMouseY(0.0)
    , mouseToggle(false)
    , Sensitivity(sensitivity)
    , MoveRatio(moveRatio)
    , MoveRatioModifier(moveRatioModifier)
    , currentMovementRatio(MoveRatio)
{}

// Interface
bool MovementSchemeFPS::InputAction(VisorTransform& transform,
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
        Vector3 lookDir = (transform.gazePoint - transform.position).NormalizeSelf();
        Vector3 side = Cross(transform.up, lookDir).NormalizeSelf();
        switch(visorAction)
        {
            // Movement
            case VisorActionType::MOVE_FORWARD:
            {
                transform.position += lookDir * static_cast<float>(currentMovementRatio);
                transform.gazePoint += lookDir * static_cast<float>(currentMovementRatio);
                break;
            }
            case VisorActionType::MOVE_LEFT:
            {
                transform.position += side * static_cast<float>(currentMovementRatio);
                transform.gazePoint += side * static_cast<float>(currentMovementRatio);
                break;
            }
            case VisorActionType::MOVE_BACKWARD:
            {
                transform.position += lookDir * static_cast<float>(-currentMovementRatio);
                transform.gazePoint += lookDir * static_cast<float>(-currentMovementRatio);
                break;
            }
            case VisorActionType::MOVE_RIGHT:
            {
                transform.position += side * static_cast<float>(-currentMovementRatio);
                transform.gazePoint += side * static_cast<float>(-currentMovementRatio);
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

bool MovementSchemeFPS::MouseMovementAction(VisorTransform& transform,
                                            double x, double y)
{
    // Check with latest recorded input
    double diffX = x - prevMouseX;
    double diffY = y - prevMouseY;

    if(mouseToggle)
    {
        // X Rotation
        Vector3 lookDir = transform.gazePoint - transform.position;
        QuatF rotateX(static_cast<float>(-diffX * Sensitivity), YAxis);
        Vector3 rotated = rotateX.ApplyRotation(lookDir);
        transform.gazePoint = transform.position + rotated;

        // Y Rotation
        lookDir = transform.gazePoint - transform.position;
        Vector3 side = Cross(transform.up, lookDir).NormalizeSelf();
        QuatF rotateY(static_cast<float>(diffY * Sensitivity), side);
        rotated = rotateY.ApplyRotation((lookDir));
        transform.gazePoint = transform.position + rotated;

        // Redefine up
        // Enforce an up vector which is orthogonal to the xz plane
        transform.up = Cross(rotated, side);
        transform.up[0] = 0.0f;
        transform.up[1] = (transform.up[1] < 0.0f) ? -1.0f : 1.0f;
        transform.up[2] = 0.0f;
    }
    prevMouseX = x;
    prevMouseY = y;
    return mouseToggle;
}

bool MovementSchemeFPS::MouseScrollAction(VisorTransform&,
                                          double, double)
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
bool MovementSchemeMaya::InputAction(VisorTransform&,
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

bool MovementSchemeMaya::MouseMovementAction(VisorTransform& transform,
                                             double x, double y)
{
    bool camChanged = false;
    // Check with latest recorded input
	float diffX = static_cast<float>(x - mouseX);
    float diffY = static_cast<float>(y - mouseY);

	if(moveMode)
	{
		// X Rotation
		Vector3f lookDir = transform.gazePoint - transform.position;
		QuatF rotateX(static_cast<float>(-diffX * Sensitivity), YAxis);
        Vector3f rotated = rotateX.ApplyRotation(lookDir);
        transform.position = transform.gazePoint - rotated;

		// Y Rotation
		lookDir = transform.gazePoint - transform.position;
        Vector3f left = Cross(transform.up, lookDir).NormalizeSelf();
        QuatF rotateY(static_cast<float>(diffY * Sensitivity), left);
		rotated = rotateY.ApplyRotation((lookDir));
        transform.position = transform.gazePoint - rotated;

		// Redefine up
		// Enforce an up vector which is orthogonal to the xz plane
        transform.up = Cross(rotated, left);
        transform.up[2] = 0.0f;
        transform.up[0] = 0.0f;
        transform.up.NormalizeSelf();
        camChanged = true;
	}
	if(translateMode)
	{
        Vector3f lookDir = transform.gazePoint - transform.position;
        Vector3f side = Cross(transform.up, lookDir).NormalizeSelf();
        transform.position += static_cast<float>(diffX * TranslateModifier) * side;
        transform.gazePoint += static_cast<float>(diffX * TranslateModifier) * side;

        transform.position += static_cast<float>(diffY * TranslateModifier) * transform.up;
        transform.gazePoint += static_cast<float>(diffY * TranslateModifier) * transform.up;
        camChanged = true;
	}

	mouseX = x;
	mouseY = y;
    return camChanged;
}

bool MovementSchemeMaya::MouseScrollAction(VisorTransform& transform,
                                           double, double y)
{
    // Zoom to the focus until some threshold
    Vector3f lookDir = transform.position - transform.gazePoint;
    lookDir *= static_cast<float>(1.0 - y * ZoomPercentage);
    if(lookDir.Length() > 0.1f)
    {
        transform.position = lookDir + transform.gazePoint;
        return true;
    }
    return false;
}