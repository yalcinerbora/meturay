#pragma once
/**


*/

#include "VisorInputI.h"
#include "Camera.h"

class MovementSchemeI;

using MovementScemeList = std::vector<std::unique_ptr<MovementSchemeI>>;

class VisorWindowInput : public VisorInputI
{
    private:
        // Key Bindings
        const MouseKeyBindings          mouseBindings;
        const KeyboardKeyBindings       keyboardBindings;

        // Movement Related
        const MovementScemeList         movementSchemes;    // List of available movers
        unsigned int                    currentMovementScheme;

        // Camera Related States
        unsigned int                    currentSceneCam;    // Currently selected scene camera
        CameraMode                      cameraMode;
        CameraPerspective               customCamera;
        bool                            lockedCamera;

        // Other States
        bool                            pauseCont;
        bool                            startStop;
        double                          deltaT;

        // Visor Callback
        VisorCallbacksI*                visorCallbacks;
        

        // Internals
        void                            ProcessInput(VisorActionType, KeyAction);

    protected:
    public:
        // Constructor & Destructor
                                VisorWindowInput(KeyboardKeyBindings&&,
                                                 MouseKeyBindings&&,
                                                 MovementScemeList&&,
                                                 const CameraPerspective& customCamera);
                                ~VisorWindowInput() = default;

        void                    ChangeDeltaT(double);
                       
        // Implementation
        void                    AttachVisorCallback(VisorCallbacksI&) override;

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
};