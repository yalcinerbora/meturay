#pragma once
/**

*/

#include "VisorInputI.h"
#include "VisorTransform.h"
#include <memory>

class MovementSchemeI;

using MovementSchemeList = std::vector<std::unique_ptr<MovementSchemeI>>;

class VisorWindowInput : public VisorInputI
{
    private:
        // Key Bindings
        const MouseKeyBindings          mouseBindings;
        const KeyboardKeyBindings       keyboardBindings;

        // Movement Related
        const MovementSchemeList        movementSchemes;    // List of available movers
        unsigned int                    currentMovementScheme;

        // Camera Related States
        unsigned int                    currentSceneCam;    // Currently selected scene camera
        CameraMode                      cameraMode;
        VisorTransform                  customTransform;
        bool                            lockedCamera;
        uint32_t                        sceneCameraCount;

        // Other States
        bool                            pauseCont;
        bool                            startStop;
        double                          deltaT;

        // Visor Callback
        VisorCallbacksI*                visorCallbacks;
        VisorI*                         visor;

        // Internals
        void                            ProcessInput(VisorActionType, KeyAction);

    protected:
    public:
        // Constructor & Destructor
                                VisorWindowInput(KeyboardKeyBindings&&,
                                                 MouseKeyBindings&&,
                                                 MovementSchemeList&&);
                                ~VisorWindowInput() = default;

        void                    ChangeDeltaT(double);

        // Implementation
        void                    AttachVisorCallback(VisorCallbacksI&) override;
        void                    SetVisor(VisorI&) override;

        void                    WindowPosChanged(int posX, int posY) override;
        void                    WindowFBChanged(int fbWidth, int fbHeight) override;
        void                    WindowSizeChanged(int width, int height) override;
        void                    WindowClosed() override;
        void                    WindowRefreshed() override;
        void                    WindowFocused(bool) override;
        void                    WindowMinimized(bool) override;

        void                    MouseScrolled(double xOffset, double yOffset) override;
        void                    MouseMoved(double x, double y) override;

        void                    KeyboardUsed(KeyboardKeyType key, KeyAction action) override;
        void                    MouseButtonUsed(MouseButtonType button, KeyAction action) override;

        void                    SetTransform(const VisorTransform&) override;
        void                    SetSceneCameraCount(uint32_t) override;
};