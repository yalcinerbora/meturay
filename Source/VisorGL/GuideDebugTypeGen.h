#pragma once

#include <memory>
#include <nlohmann/json_fwd.hpp>

#include "GDebugRendererI.h"

class TextureGL;

using DebugRendererPtr = std::unique_ptr<GDebugRendererI>;

template<class GDebugRendererI>
using GDBRendererGenFunc = GDebugRendererI * (*)(const nlohmann::json& config,
                                                 const TextureGL& gradientTexture,
                                                 const std::string& configPath,
                                                 uint32_t depthCount);

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
                                const TextureGL& gradientTexture,
                                const std::string& configPath,
                                uint32_t depthCount)
    {
        GDebugRendererI* renderer = gFunc(config, gradientTexture,
                                          configPath, depthCount);
        return DebugRendererPtr(renderer);
    }
};

template <class Base, class GDBRenderer>
Base* GDBRendererConstruct(const nlohmann::json& config,
                           const TextureGL& gradientTexture,
                           const std::string& configPath,
                           uint32_t depthCount)
{
    return new GDBRenderer(config, gradientTexture, configPath, depthCount);
}