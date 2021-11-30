#pragma once

#include "RayLib/Vector.h"

struct AnalyticData;
struct SceneAnalyticData;

class MainStatusBar
{

    private:
        static constexpr const char* RENDERING  = "Rendering";
        static constexpr const char* PAUSED     = "Paused";
        static constexpr const char* STOPPED    = "Stopped";

    protected:
    public:

        void        Render(const AnalyticData&,
                           const SceneAnalyticData&,
                           const Vector2i& iSize);
};