#include "VisorWindowInput.h"
#include "MovementSchemeI.h"

#include "Vector.h"
#include "VisorTransform.h"
#include "VisorCallbacksI.h"
#include "Quaternion.h"
#include "Log.h"
#include "VisorI.h"

void VisorWindowInput::ProcessInput(VisorActionType vAction, KeyAction action)
{
    switch(vAction)
    {
        case VisorActionType::MOVE_TYPE_NEXT:
        {
            if(action != KeyAction::RELEASED) break;
            currentMovementScheme = (currentMovementScheme + 1) % movementSchemes.size();
            break;
        }
        case VisorActionType::MOVE_TYPE_PREV:
        {
            if(action != KeyAction::RELEASED) break;
            currentMovementScheme = (currentMovementScheme - 1) % movementSchemes.size();
            break;
        }
        case VisorActionType::TOGGLE_CUSTOM_SCENE_CAMERA:
        {
            if(action != KeyAction::RELEASED) break;
            cameraMode = (cameraMode == CameraMode::CUSTOM_CAM)
                            ? CameraMode::SCENE_CAM
                            : CameraMode::CUSTOM_CAM;
            break;
        }
        case VisorActionType::LOCK_UNLOCK_CAMERA:
        {
            if(action != KeyAction::RELEASED) break;
            lockedCamera = !lockedCamera;
            break;
        }
        case VisorActionType::SCENE_CAM_NEXT:
        case VisorActionType::SCENE_CAM_PREV:
        {
            if(cameraMode == CameraMode::CUSTOM_CAM ||
               lockedCamera || (action != KeyAction::RELEASED))
                break;

            currentSceneCam = (vAction == VisorActionType::SCENE_CAM_NEXT)
                                ? currentSceneCam + 1
                                : currentSceneCam - 1;
            currentSceneCam %= sceneCameraCount;
            visorCallbacks->ChangeCamera(currentSceneCam);
            break;
        }
        case VisorActionType::PRINT_CUSTOM_CAMERA:
        {
            if(action != KeyAction::RELEASED) break;

            std::string tAsString = VisorTransformToString(customTransform);
            METU_LOG(tAsString);
            break;
        }
        case VisorActionType::START_STOP_TRACE:
        {
            if(action != KeyAction::RELEASED) break;

            visorCallbacks->StartStopTrace(startStop);
            startStop = !startStop;
            break;
        }
        case VisorActionType::PAUSE_CONT_TRACE:
        {
            if(action != KeyAction::RELEASED) break;

            pauseCont = !pauseCont;
            visorCallbacks->PauseContTrace(pauseCont);
            break;
        }
        case VisorActionType::FRAME_NEXT:
        {
            if(action != KeyAction::RELEASED) break;

            visorCallbacks->IncreaseTime(deltaT);
            break;
        }
        case VisorActionType::FRAME_PREV:
        {
            if(action != KeyAction::RELEASED) break;

            visorCallbacks->DecreaseTime(deltaT);
            break;
        }
        case VisorActionType::SAVE_IMAGE:
        {
            if(action != KeyAction::RELEASED) break;

            if(visor) visor->SaveImage(false);
            break;
        }
        case VisorActionType::SAVE_IMAGE_HDR:
        {
            if(action != KeyAction::RELEASED) break;

            if(visor) visor->SaveImage(true);
            break;
        }
        case VisorActionType::CLOSE:
        {
            if(action != KeyAction::RELEASED) break;

            visorCallbacks->WindowCloseAction();
            break;
        }
        default:
            break;
    }
}

VisorWindowInput::VisorWindowInput(KeyboardKeyBindings&& keyBinds,
                                   MouseKeyBindings&& mouseBinds,
                                   MovementSchemeList&& movementSchemes)
    : mouseBindings(std::move(mouseBinds))
    , keyboardBindings(std::move(keyBinds))
    , movementSchemes(std::move(movementSchemes))
    , currentMovementScheme(0)
    , currentSceneCam(0)
    , cameraMode(CameraMode::SCENE_CAM)
    , lockedCamera(false)
    , sceneCameraCount(0)
    , pauseCont(false)
    , startStop(false)
    , deltaT(1.0)
    , visorCallbacks(nullptr)
    , visor(nullptr)
{}

void VisorWindowInput::ChangeDeltaT(double dT)
{
    deltaT = dT;
}

void VisorWindowInput::AttachVisorCallback(VisorCallbacksI& vc)
{
    visorCallbacks = &vc;
}

VisorCallbacksI* VisorWindowInput::CurrentVisorCallback() const
{
    return visorCallbacks;
}

void VisorWindowInput::SetVisor(VisorI& v)
{
    visor = &v;
}

void VisorWindowInput::WindowPosChanged(int, int)
{}

void VisorWindowInput::WindowFBChanged(int, int)
{}

void VisorWindowInput::WindowSizeChanged(int, int)
{}

void VisorWindowInput::WindowClosed()
{
    visorCallbacks->WindowCloseAction();
}

void VisorWindowInput::WindowRefreshed()
{}

void VisorWindowInput::WindowFocused(bool)
{}

void VisorWindowInput::WindowMinimized(bool minimized)
{
    visorCallbacks->WindowMinimizeAction(minimized);
}

void VisorWindowInput::MouseScrolled(double xOffset, double yOffset)
{
    if(cameraMode == CameraMode::CUSTOM_CAM && !lockedCamera)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);

        if(currentScheme.MouseScrollAction(customTransform, xOffset, yOffset))
            visorCallbacks->ChangeCamera(customTransform);
    }
}

void VisorWindowInput::MouseMoved(double x, double y)
{
    if(cameraMode == CameraMode::CUSTOM_CAM && !lockedCamera)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);

        if(currentScheme.MouseMovementAction(customTransform, x, y))
            visorCallbacks->ChangeCamera(customTransform);
    }
}

void VisorWindowInput::KeyboardUsed(KeyboardKeyType key,
                                    KeyAction action)
{
    // Find an action if avail
    KeyboardKeyBindings::const_iterator i;
    if((i = keyboardBindings.find(key)) == keyboardBindings.cend()) return;
    VisorActionType visorAction = i->second;

    // Do custom cam
    if(cameraMode == CameraMode::CUSTOM_CAM && !lockedCamera)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);
        if(currentScheme.InputAction(customTransform, visorAction, action))
            visorCallbacks->ChangeCamera(customTransform);
    }

    // Do other
    ProcessInput(visorAction, action);
}

void VisorWindowInput::MouseButtonUsed(MouseButtonType button, KeyAction action)
{
    // Find an action if avail
    MouseKeyBindings::const_iterator i;
    if((i = mouseBindings.find(button)) == mouseBindings.cend()) return;
    VisorActionType visorAction = i->second;

    // Do Custom Camera
    if(cameraMode == CameraMode::CUSTOM_CAM && !lockedCamera)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);
        if(currentScheme.InputAction(customTransform, visorAction, action))
            visorCallbacks->ChangeCamera(customTransform);
    }

    // Do Other
    ProcessInput(visorAction, action);
}

void VisorWindowInput::SetTransform(const VisorTransform& t)
{
    if(cameraMode == CameraMode::SCENE_CAM)
        customTransform = t;
}

void VisorWindowInput::SetSceneCameraCount(uint32_t camCount)
{
    sceneCameraCount = camCount;

    if(currentSceneCam > sceneCameraCount)
        currentSceneCam = 0;
}