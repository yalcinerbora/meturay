#pragma once

#include <map>

#include "SurfaceLoaderGeneratorI.h"
#include "SurfaceLoaderPoolI.h"
#include "SharedLib.h"

using SurfaceLoaderPoolPtr = SharedLibPtr<SurfaceLoaderPoolI>;


class SurfaceLoaderGenerator : public SurfaceLoaderGeneratorI
{
    private:
        // Shared Libraries That are Loaded
        std::map<std::string, SharedLib>                openedLibs;
        // Loaded Surface Loader Pools
        std::map<PoolKey, SurfaceLoaderPoolPtr>         generatedPools;

        // Surface Loader Generators (Combined from all dlls)
        std::map<std::string, SurfaceLoaderGen>         loaderGenerators;

    protected:


    public:
        // Constructors & Destructor
                                    SurfaceLoaderGenerator();
                                    ~SurfaceLoaderGenerator() = default;

        // Interface
        SceneError                  GenerateSurfaceLoader(SharedLibPtr<SurfaceLoaderI>&,
                                                          const std::string& scenePath,
                                                          const SceneNodeI& properties,
                                                          double time = 0.0) const override;
        //
        DLLError                    IncludeLoadersFromDLL(const std::string& libName,
                                                          const std::string& regex,
                                                          const SharedLibArgs& mangledName) override;
};
