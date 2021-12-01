#pragma once
/**

*/

#include "RayLib/VisorInputI.h"
#include "RayLib/VisorTransform.h"
#include "RayLib/TracerStructs.h"
#include "RayLib/MovementSchemeI.h"
#include <memory>

using MovementSchemeList = std::vector<std::unique_ptr<MovementSchemeI>>;
class ImageSaverI;

class VisorWindowInput : public VisorInputI
{
    private:
        // Internals
        void            ProcessInput(VisorActionType, KeyAction);
        void            StartStopRunState();
        void            PauseContRunState();

    protected:
        VisorCallbacksI&            visorCallbacks;
        // Camera Related States
        unsigned int                currentSceneCam;    // Currently selected scene camera
        TracerCameraMode            cameraMode;
        VisorTransform              customTransform;
        bool                        lockedCamera;
        uint32_t                    sceneCameraCount;
        // Other States
        TracerRunState              tracerRunState;
        double                      deltaT;
        bool&                       isWindowOpen;
        Vector2i&                   windowSize;
        Vector2i&                   viewportSize;
        // Image Save Function Interface
        ImageSaverI&                saver;

        // Binding Related
        const MouseKeyBindings      mouseBindings;
        const KeyboardKeyBindings   keyboardBindings;
        // Movement Related
        const MovementSchemeList    movementSchemes;        // List of available movers
        unsigned int                currentMovementScheme;

    public:
        // Constructor & Destructor
                                VisorWindowInput(VisorCallbacksI&,
                                                 bool& isWindowOpen,
                                                 Vector2i& windowSize,
                                                 Vector2i& viewportSize,
                                                 ImageSaverI&,
                                                 const KeyboardKeyBindings&,
                                                 const MouseKeyBindings&,
                                                 MovementSchemeList&&);
                                ~VisorWindowInput() = default;
        // Implementation
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

        // Setters
        void                    SetTransform(const VisorTransform&) override;
        void                    SetSceneCameraCount(uint32_t) override;
        void                    SetSceneAnalyticData(const SceneAnalyticData&) override;
        void                    SetTracerAnalyticData(const TracerAnalyticData&) override;
        void                    SetTracerOptions(const TracerOptions&) override;
        void                    SetTracerParams(const TracerParameters&) override;

        void                    RenderGUI() override;
};