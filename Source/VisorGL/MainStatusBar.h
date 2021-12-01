#pragma once

#include "RayLib/Vector.h"

struct AnalyticData;
struct SceneAnalyticData;

class MainStatusBar
{
    public:
        enum RenderState
        {
            RUNNING,
            PAUSED,
            STOPPED
        };

    private:
        static constexpr const char* RENDERING_NAME = "Rendering";
        static constexpr const char* PAUSED_NAME    = "PAUSED";
        static constexpr const char* STOPPED_NAME   = "STOPPED";

        bool        paused;
        bool        running;
        bool        stopped;

    protected:
    public:
        // Constructors & Destructor
                    MainStatusBar();
                    ~MainStatusBar() = default;

        void        Render(const AnalyticData&,
                           const SceneAnalyticData&,
                           const Vector2i& iSize);

        void        SetState(RenderState);
};