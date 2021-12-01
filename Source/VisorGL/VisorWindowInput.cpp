#include "VisorWindowInput.h"

#include "RayLib/MovementSchemeI.h"
#include "RayLib/Vector.h"
#include "RayLib/VisorTransform.h"
#include "RayLib/VisorCallbacksI.h"
#include "RayLib/Quaternion.h"
#include "RayLib/Log.h"
#include "RayLib/VisorI.h"

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
            cameraMode = (cameraMode == TracerCameraMode::CUSTOM_CAM)
                            ? TracerCameraMode::SCENE_CAM
                            : TracerCameraMode::CUSTOM_CAM;
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
            if(cameraMode == TracerCameraMode::CUSTOM_CAM ||
               lockedCamera || (action != KeyAction::RELEASED))
                break;

            currentSceneCam = (vAction == VisorActionType::SCENE_CAM_NEXT)
                                ? currentSceneCam + 1
                                : currentSceneCam - 1;
            currentSceneCam %= sceneCameraCount;
            visorCallbacks.ChangeCamera(currentSceneCam);
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
            StartStopRunState();
            break;
        }
        case VisorActionType::PAUSE_CONT_TRACE:
        {
            if(action != KeyAction::RELEASED) break;
            PauseContRunState();
            break;
        }
        case VisorActionType::FRAME_NEXT:
        {
            if(action != KeyAction::RELEASED) break;

            visorCallbacks.IncreaseTime(deltaT);
            break;
        }
        case VisorActionType::FRAME_PREV:
        {
            if(action != KeyAction::RELEASED) break;

            visorCallbacks.DecreaseTime(deltaT);
            break;
        }
        case VisorActionType::SAVE_IMAGE:
        {
            if(action != KeyAction::RELEASED) break;

            saver.SaveImage(false);
            break;
        }
        case VisorActionType::SAVE_IMAGE_HDR:
        {
            if(action != KeyAction::RELEASED) break;

            saver.SaveImage(true);
            break;
        }
        case VisorActionType::CLOSE:
        {
            if(action != KeyAction::RELEASED) break;

            visorCallbacks.WindowCloseAction();
            break;
        }
        default:
            break;
    }
}

void VisorWindowInput::StartStopRunState()
{
    if(tracerRunState == TracerRunState::RUNNING ||
       tracerRunState == TracerRunState::PAUSED)
    {
        tracerRunState = TracerRunState::STOPPED;
        visorCallbacks.StartStopTrace(false);
    }
    else if(tracerRunState == TracerRunState::STOPPED)
    {
        tracerRunState = TracerRunState::RUNNING;
        visorCallbacks.StartStopTrace(true);
    }
}

void VisorWindowInput::PauseContRunState()
{
    if(tracerRunState == TracerRunState::RUNNING)
    {
        tracerRunState = TracerRunState::PAUSED;
        visorCallbacks.PauseContTrace(true);
    }
    else if(tracerRunState == TracerRunState::PAUSED)
    {
        tracerRunState = TracerRunState::RUNNING;
        visorCallbacks.PauseContTrace(false);
    }
    else return;
}

VisorWindowInput::VisorWindowInput(VisorCallbacksI& cb,
                                   bool& isWindowOpen,
                                   Vector2i& windowSize,
                                   Vector2i& viewportSize,
                                   ImageSaverI& saver,
                                   const KeyboardKeyBindings& keyBinds,
                                   const MouseKeyBindings& mouseBinds,
                                   MovementSchemeList&& movementSchemes)
    : visorCallbacks(cb)
    , currentSceneCam(0)
    , cameraMode(TracerCameraMode::SCENE_CAM)
    , lockedCamera(false)
    , sceneCameraCount(0)
    , tracerRunState(TracerRunState::RUNNING)
    , deltaT(1.0)
    , isWindowOpen(isWindowOpen)
    , windowSize(windowSize)
    , viewportSize(viewportSize)
    , saver(saver)
    , mouseBindings(mouseBinds)
    , keyboardBindings(keyBinds)
    , movementSchemes(std::move(movementSchemes))
    , currentMovementScheme(0)
{}


void VisorWindowInput::WindowPosChanged(int, int)
{}

void VisorWindowInput::WindowFBChanged(int x, int y)
{
    viewportSize = Vector2i(x, y);
}

void VisorWindowInput::WindowSizeChanged(int x, int y)
{
    windowSize = Vector2i(x, y);
}

void VisorWindowInput::WindowClosed()
{
    isWindowOpen = false;
    visorCallbacks.WindowCloseAction();
}

void VisorWindowInput::WindowRefreshed()
{}

void VisorWindowInput::WindowFocused(bool)
{}

void VisorWindowInput::WindowMinimized(bool minimized)
{
    visorCallbacks.WindowMinimizeAction(minimized);
}

void VisorWindowInput::MouseScrolled(double xOffset, double yOffset)
{
    if(cameraMode == TracerCameraMode::CUSTOM_CAM && !lockedCamera)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);

        if(currentScheme.MouseScrollAction(customTransform, xOffset, yOffset))
            visorCallbacks.ChangeCamera(customTransform);
    }
}

void VisorWindowInput::MouseMoved(double x, double y)
{
    if(cameraMode == TracerCameraMode::CUSTOM_CAM && !lockedCamera)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);

        if(currentScheme.MouseMovementAction(customTransform, x, y))
            visorCallbacks.ChangeCamera(customTransform);
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
    if(cameraMode == TracerCameraMode::CUSTOM_CAM && !lockedCamera)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);
        if(currentScheme.InputAction(customTransform, visorAction, action))
            visorCallbacks.ChangeCamera(customTransform);
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
    if(cameraMode == TracerCameraMode::CUSTOM_CAM && !lockedCamera)
    {
        MovementSchemeI& currentScheme = *(movementSchemes[currentMovementScheme]);
        if(currentScheme.InputAction(customTransform, visorAction, action))
            visorCallbacks.ChangeCamera(customTransform);
    }

    // Do Other
    ProcessInput(visorAction, action);
}

void VisorWindowInput::SetTransform(const VisorTransform& t)
{
    if(cameraMode == TracerCameraMode::SCENE_CAM)
        customTransform = t;
}

void VisorWindowInput::SetSceneCameraCount(uint32_t camCount)
{
    sceneCameraCount = camCount;

    if(currentSceneCam > sceneCameraCount)
        currentSceneCam = 0;
}

void VisorWindowInput::SetSceneAnalyticData(const SceneAnalyticData&) {}
void VisorWindowInput::SetTracerAnalyticData(const TracerAnalyticData&) {}
void VisorWindowInput::SetTracerOptions(const TracerOptions&) {}
void VisorWindowInput::SetTracerParams(const TracerParameters&) {}
void VisorWindowInput::RenderGUI() {}