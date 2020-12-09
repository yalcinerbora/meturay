#pragma once

#include <string>
#include <memory>

#include "TracerStructs.h"

#include "RayLib/SharedLib.h"
#include "RayLib/GPUSceneI.h"

struct TracerError;
struct SceneError;

class SurfaceLoaderGenerator;
class TracerOptions;

enum class ScenePartitionerType
{
    SINGLE_GPU,
    // Not yet implemented
    MULTI_GPU_MATERIAL
};

struct SurfaceLoaderSharedLib
{
    const std::string       libName;
    const std::string       regex;
    const SharedLibArgs     mangledName;
};

class TracerSystemI
{

    public:
        virtual                         ~TracerSystemI() = default;

        // Interface
        virtual TracerError             Initialize(const std::vector<SurfaceLoaderSharedLib>&,
                                                   ScenePartitionerType) = 0;

        virtual void                    GenerateScene(GPUSceneI*&,
                                                      const std::u8string& scenePath) = 0;
        virtual TracerError             GenerateTracer(GPUTracerPtr&,
                                                       const TracerParameters&,
                                                       const TracerOptions&,
                                                       const std::string& tracerType) = 0;
};