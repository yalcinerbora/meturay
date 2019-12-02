#pragma once

#include <string>

#include "ObjectFuncDefinitions.h"

struct DLLError;
struct SceneError;
struct SharedLibArgs;

class SharedLib;
class SurfaceLoaderI;
class SceneNodeI;

class SurfaceLoaderGeneratorI
{
    public:
        virtual                 ~SurfaceLoaderGeneratorI() = default;

        virtual SceneError      GenerateSurfaceLoader(SharedLibPtr<SurfaceLoaderI>&,
                                                      const std::string& scenePath,
                                                      const SceneNodeI& properties,
                                                      double time = 0.0) const = 0;

        virtual DLLError        IncludeLoadersFromDLL(const std::string& libName,
                                                      const std::string& regex,
                                                      const SharedLibArgs& mangledName) = 0;
};