#pragma once

#include "RayLib/Vector.h"

struct TracerAnalyticData;
struct SceneAnalyticData;
class VisorCallbacksI;

enum class TracerRunState;

class MainStatusBar
{
    private:
        static constexpr const char* RENDERING_NAME = "Rendering";
        static constexpr const char* PAUSED_NAME    = "PAUSED";
        static constexpr const char* STOPPED_NAME   = "STOPPED";

        bool                    paused;
        bool                    running;
        bool                    stopped;

        static TracerRunState   DetermineTracerState(bool stopToggle,
                                                     bool runToggle,
                                                     bool pauseToggle);
        static void             SetButtonState(bool& stopToggle,
                                               bool& runToggle,
                                               bool& pauseToggle,
                                               TracerRunState);

    protected:
    public:
        // Constructors & Destructor
                    MainStatusBar();
                    ~MainStatusBar() = default;

        void        Render(VisorCallbacksI&,
                           TracerRunState& rs,
                           const TracerAnalyticData&,
                           const SceneAnalyticData&,
                           const Vector2i& iSize);
};