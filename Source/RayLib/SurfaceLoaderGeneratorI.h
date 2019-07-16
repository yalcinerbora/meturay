#pragma once

#include <string>

#include "ObjectFuncDefinitions.h"

struct SceneError;

class SharedLib;
class SurfaceLoaderI;
class SceneNodeI;

class SurfaceLoaderGeneratorI
{
    public:
        virtual                 ~SurfaceLoaderGeneratorI() = default;

        virtual SceneError      GenerateSurfaceLoader(SharedLibPtr<SurfaceLoaderI>&,
                                                      const SceneNodeI& properties,
                                                      double time = 0.0) const = 0;

        virtual SceneError      IncludeLoadersFromDLL(const SharedLib&,
                                                      const std::string& mangledName = "\0") const = 0;
};