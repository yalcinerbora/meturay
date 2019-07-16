#pragma once

#include <map>

#include "SurfaceLoaderGeneratorI.h"

template<class Loader>
using SurfaceLoaderGeneratorFunc = Loader* (*)(const SceneNodeI&,
                                               double time);

using SurfaceLoaderPtr = SharedLibPtr<SurfaceLoaderI>;

class SurfaceLoaderGen
{
    private:
        SurfaceLoaderGeneratorFunc<SurfaceLoaderI>        gFunc;
        ObjDestroyerFunc<SurfaceLoaderI>                  dFunc;

    public:
        // Constructor & Destructor
        SurfaceLoaderGen(SurfaceLoaderGeneratorFunc<SurfaceLoaderI> g,
                         ObjDestroyerFunc<SurfaceLoaderI> d)
            : gFunc(g)
            , dFunc(d)
        {}

        SurfaceLoaderPtr operator()(const SceneNodeI& n,
                                    double time) const
        {
            SurfaceLoaderI* loader = gFunc(n, time);
            return SurfaceLoaderPtr(loader, dFunc);
        }
};

class SurfaceLoaderGenerator : public SurfaceLoaderGeneratorI
{
    private:
    protected:
        std::map<std::string, SurfaceLoaderGen>     loaderGenerators;

    public:
        // Constructors & Destructor
                                                    SurfaceLoaderGenerator();
                                                    ~SurfaceLoaderGenerator() = default;

        // Interface
        SceneError                                  GenerateSurfaceLoader(SharedLibPtr<SurfaceLoaderI>&,
                                                                          const SceneNodeI& properties,
                                                                          double time = 0.0) const override;
        SceneError                                  IncludeLoadersFromDLL(const SharedLib&,
                                                                          const std::string& mangledName = "\0") const override;
};
