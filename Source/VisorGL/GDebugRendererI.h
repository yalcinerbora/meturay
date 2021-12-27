#pragma once

#include <vector>
#include <RayLib/Vector.h>

class   TextureGL;
struct  ImVec2;

class GDebugRendererI
{
    public:
        virtual                     ~GDebugRendererI() = default;

        // Interface
        virtual void                RenderSpatial(TextureGL&, uint32_t depth,
                                                  const std::vector<Vector3f>& worldPositions) = 0;
        virtual void                UpdateDirectional(const Vector3f& worldPos,
                                                      bool doLogScale,
                                                      uint32_t depth) = 0;

        virtual bool                RenderGUI(bool& overlayCheckboxChanged,
                                              bool& overlayValue,
                                              const ImVec2& windowSize) = 0;
};