#pragma once

#include <string>
#include <vector>
#include <memory>

#include "VisorInputStructs.h"

class TracerOptions;
class MovementSchemeI;

struct VisorOptions;
struct TracerParameters;
struct SharedLibArgs;
struct SurfaceLoaderSharedLib;

enum class ScenePartitionerType;

using MovementSchemeList = std::vector<std::unique_ptr<MovementSchemeI>>;

namespace ConfigParser
{
    bool ParseVisorOptions(// Visor Input Related
                           KeyboardKeyBindings& keyBindings,
                           MouseKeyBindings& mouseBindings,
                           MovementSchemeList& movementSchemes,
                           // Visor Related
                           VisorOptions& opts,
                           // Visor DLL Related
                           std::string& visorDLLName,
                           SharedLibArgs& dllEntryPointName,
                           //
                           const std::string& configFileName);

    bool ParseTracerOptions(// Tracer Related
                            TracerOptions& tracerOptions,
                            TracerParameters& tracerParameters,
                            std::string& tracerTypeName,
                            // Tracer DLL Related
                            std::string& tracerDLLName,
                            SharedLibArgs& dllEntryPointName,
                            // Misc
                            std::vector<SurfaceLoaderSharedLib>& surfaceLoaderDLLs,
                            ScenePartitionerType& gpuUsage,
                            //
                            const std::string& configFileName);
}