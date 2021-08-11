#pragma once

#include <RayLib/Vector.h>

class TextureGL;

class GDebugRendererI
{
    public:
        virtual                     ~GDebugRendererI() = default;

        // Interface
        virtual void                RenderSpatial(TextureGL&, uint32_t depth) = 0;
        virtual void                RenderDirectional(TextureGL&, const Vector3f& worldPos,
                                                      uint32_t depth) = 0;

        virtual const std::string&  Name() const = 0;
};