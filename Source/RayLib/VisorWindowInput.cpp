#include "VisorWindowInput.h"
#include "MovementSchemeI.h"

#include "Vector.h"
#include "VisorCamera.h"
#include "VisorCallbacksI.h"
#include "Quaternion.h"

void VisorWindowInput::ProcessInput(VisorActionType vAction, KeyAction action)
{
    switch(vAction)
    {
        case VisorActionType::MOVE_TYPE_NEXT:
        {
            if(lockedCamera) break;
            currentMovementScheme = (currentMovementScheme + 1) % movementSchemes.size();
            break;
        }
        case VisorActionType::MOVE_TYPE_PREV:
        {
            if(lockedCamera) break;
            currentMovementScheme = (currentMovementScheme - 1) % movementSchemes.size();
            break;
        }
        case VisorActionType::TOGGLE_CUSTOM_SCENE_CAMERA:
        {
            if(lockedCamera) break;
            cameraMode = (cameraMode == CameraMode::CUSTOM_CAM) ? CameraMode::SCENE_CAM : CameraMode::CUSTOM_CAM;
            break;
        }
        case VisorActionType::LOCK_UNLOCK_CAMERA:
        {
            lockedCamera = !lockedCamera;
            break;
        }
        case VisorActionType::SCENE_CAM_NEXT:
        {
            if(lockedCamera) break;

            visorCallbacks->ChangeCamera(currentSceneCam);
            break;
        }
        case VisorActionType::SCENE_CAM_PREV:
        {
            if(lockedCamera) break;
            visorCallbacks->ChangeCamera(currentSceneCam);
            break;
        }
        case VisorActionType::START_STOP_TRACE:
        {
            visorCallbacks->StartStopTrace(startStop);
            startStop = !startStop;
            break;
        }
        case VisorActionType::PAUSE_CONT_TRACE:
        {
            visorCallbacks->PauseContTrace(pauseCont);
            pauseCont = !pauseCont;
            break;
        }
        case VisorActionType::FRAME_NEXT:
        {
            visorCallbacks->IncreaseTime(deltaT);
            break;
        }
        case VisorActionType::FRAME_PREV:
        {
            visorCallbacks->DecreaseTime(deltaT);
            break;
        }
        case VisorActionType::SAVE_IMAGE:
        {
            break;
        }
        case VisorActionType::CLOSE:
        {
            visorCallbacks->WindowCloseAction();
            break;
        }
        default:
            break;
    }
}

VisorWindowInput::VisorWindowInput(KeyboardKeyBindings&& keyBinds,
                                   MouseKeyBindings&& mouseBinds,
                                   MovementScemeList&& movementSchemes,
                                   const VisorCamera& customCamera)
    : mouseBindings(std::move(mouseBinds))
    , keyboardBindings(std::move(keyBinds))
    , movementSchemes(std::move(movementSchemes))
    , currentMovementScheme(0)
    , currentSceneCam(0)
    , cameraMode(CameraMode::SCENE_CAM)
    , customCamera(customCamera)
    , lockedCamera(false)
    , pauseCont(false)
    , startStop(false)
    , deltaT(1.0)
    , visorCallbacks(nullptr)
{}

void VisorWindowInput::ChangeDeltaT(double dT)
{
    deltaT = dT;
}

void VisorWindowInput::AttachVisorCallback(VisorCallbacksI& vc)
{
    visorCallbacks = &vc;
}

void VisorWindowInput::WindowPosChanged(int posX, int posY)
{}

void VisorWindowInput::WindowFBChanged(int fbWidth, int fbHeight)
{}

void VisorWindowInput::WindowSizeChanged(int width, int height)
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
    if(cameraMode == CameraMode::CUSTOM_CAM)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);

        if(currentScheme.MouseScrollAction(customCamera, xOffset, yOffset))
            visorCallbacks->ChangeCamera(customCamera);
    }
}

void VisorWindowInput::MouseMoved(double x, double y)
{
    if(cameraMode == CameraMode::CUSTOM_CAM)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);

        if(currentScheme.MouseMovementAction(customCamera, x, y))
            visorCallbacks->ChangeCamera(customCamera);
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
    if(cameraMode == CameraMode::CUSTOM_CAM)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);
        if(currentScheme.InputAction(customCamera, visorAction, action))
            visorCallbacks->ChangeCamera(customCamera);
    }

    // Do other
    ProcessInput(visorAction, action);
}

void VisorWindowInput::MouseButtonUsed(MouseButtonType button, KeyAction action)
{
    // Find an action if avail
    MouseKeyBindings::const_iterator i;
    if((i = mouseBindings.find(button)) != mouseBindings.cend()) return;
    VisorActionType visorAction = i->second;

    // Do Custom Camera
    if(cameraMode == CameraMode::CUSTOM_CAM)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);
        if(currentScheme.InputAction(customCamera, visorAction, action))
            visorCallbacks->ChangeCamera(customCamera);
    }

    // Do Other
    ProcessInput(visorAction, action);
}