#pragma once

#include <memory>
#include <nlohmann/json_fwd.hpp>

#include "GDebugRendererI.h"

class TextureGL;

using DebugRendererPtr = std::unique_ptr<GDebugRendererI>;

template<class GDebugRendererI>
using GDBRendererGenFunc = GDebugRendererI * (*)(const nlohmann::json& config,
                                                 const TextureGL& gradientTexture);

class GDBRendererGen
{
    private:
    GDBRendererGenFunc<GDebugRendererI>   gFunc;

    public:
    // Constructor & Destructor
    GDBRendererGen(GDBRendererGenFunc<GDebugRendererI> g)
        : gFunc(g)
    {}

    DebugRendererPtr operator()(const nlohmann::json& config,
                                const TextureGL& gradientTexture)
    {
        GDebugRendererI* renderer = gFunc(config, gradientTexture);
        return DebugRendererPtr(renderer);
    }
};

template <class Base, class GDBRenderer>
Base* GDBRendererConstruct(const nlohmann::json& config,
                           const TextureGL& gradientTexture)
{
    return new GDBRenderer(config, gradientTexture);
}