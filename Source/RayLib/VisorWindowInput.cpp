#include "VisorWindowInput.h"
#include "RayLib/Vector.h"
#include "RayLib/Camera.h"
#include "RayLib/VisorCallbacksI.h"
#include "RayLib/Quaternion.h"

VisorWindowInput::VisorWindowInput(double sensitivity,
                                   double moveRatio,
                                   double moveRatioModifier,
                                   //
                                   const CameraPerspective& initialCamera)
    : Sensitivity(sensitivity)
    , MoveRatio(moveRatio)
    , MoveRatioModifier(moveRatioModifier)
    , fpsMode(false)
    , camera(initialCamera)
    , visorCallbacks(nullptr)
    , mouseX(0.0f)
    , mouseY(0.0f)
    , animationFPS(1.0)
{}

void VisorWindowInput::ChangeAnimationFPS(double fps)
{
    animationFPS = fps;
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
    // TODO: Do zoom
}

void VisorWindowInput::MouseMoved(double x, double y)
{
    // Check with latest recorded input
    double diffX = x - mouseX;
    double diffY = y - mouseY;

    if(fpsMode)
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

        visorCallbacks->ChangeCamera(camera);
    }
    mouseX = x;
    mouseY = y;
}

void VisorWindowInput::KeyboardUsed(KeyboardKeyType key,
                                    KeyAction action)
{
    // Shift modifier
    double currentRatio = 0.0;
    if(action == KeyAction::PRESSED && key == KeyboardKeyType::LEFT_SHIFT)
    {
        currentRatio = MoveRatio * MoveRatioModifier;
    }
    else if(action == KeyAction::RELEASED  && key == KeyboardKeyType::LEFT_SHIFT)
    {
        currentRatio = MoveRatio;
    }

    bool camChanged = false;
    if(!(action == KeyAction::RELEASED))
    {
        Vector3 lookDir = (camera.gazePoint - camera.position).NormalizeSelf();
        Vector3 side = Cross(camera.up, lookDir).NormalizeSelf();
        switch(key)
        {
            // Movement
            case KeyboardKeyType::W:
            {
                camera.position += lookDir * static_cast<float>(currentRatio);
                camera.gazePoint += lookDir * static_cast<float>(currentRatio);
                camChanged = true;
                break;
            }
            case KeyboardKeyType::A:
            {
                camera.position += side * static_cast<float>(currentRatio);
                camera.gazePoint += side * static_cast<float>(currentRatio);
                camChanged = true;
                break;
            }
            case KeyboardKeyType::S:
            {
                camera.position += lookDir * static_cast<float>(-currentRatio);
                camera.gazePoint += lookDir * static_cast<float>(-currentRatio);
                camChanged = true;
                break;
            }
            case KeyboardKeyType::D:
            {
                camera.position += side * static_cast<float>(-currentRatio);
                camera.gazePoint += side * static_cast<float>(-currentRatio);
                camChanged = true;
                break;
            }
            // Save functionality
            case KeyboardKeyType::ENTER:
            {
                //
                break;
            }

            // Next-Previous frame
            case KeyboardKeyType::RIGHT:
            {
                visorCallbacks->IncreaseTime(1.0f / static_cast<float>(animationFPS));
                break;
            }
            case KeyboardKeyType::LEFT:
            {
                visorCallbacks->DecreaseTime(1.0f / static_cast<float>(animationFPS));
                break;
            }
            // Pause cont. Tracer
            case  KeyboardKeyType::P:
            {
                //visorCallbacks->PauseContTrace(pause);
                //pause = !pause;
                break;
            }

            default:
                // Do nothing on other keys
                break;
        }
    }

    // Check if Camera Changed
    if(camChanged) visorCallbacks->ChangeCamera(camera);
}

void VisorWindowInput::MouseButtonUsed(MouseButtonType button, KeyAction action)
{
    switch(button)
    {
        case MouseButtonType::LEFT:
        case MouseButtonType::RIGHT:
            fpsMode = (action == KeyAction::RELEASED) ? false : true;
            break;
    }
}