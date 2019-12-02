#pragma once

#include "SurfaceLoaderI.h"
#include "ObjectFuncDefinitions.h"
#include <regex>
#include <map>
#include <string>

template<class Loader>
using SurfaceLoaderGeneratorFunc = Loader* (*)(const std::string& scenePath, 
                                               const SceneNodeI&,
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

        SurfaceLoaderPtr operator()(const std::string& scenePath,
                                    const SceneNodeI& n,
                                    double time) const
        {
            SurfaceLoaderI* loader = gFunc(scenePath, n, time);
            return SurfaceLoaderPtr(loader, dFunc);
        }
};

class SurfaceLoaderPoolI
{
    protected:
        std::map<std::string, SurfaceLoaderGen>   surfaceLoaderGenerators;

    public:
        static constexpr const char* DefaultConstructorName = "GenSurfaceLoaderPool";
        static constexpr const char* DefaultDestructorName = "DelSurfaceLoaderPool";

        virtual                     ~SurfaceLoaderPoolI() = default;

        virtual std::map<std::string, SurfaceLoaderGen> SurfaceLoaders(const std::string regex = ".*") const;
};

inline std::map<std::string, SurfaceLoaderGen> SurfaceLoaderPoolI::SurfaceLoaders(const std::string regex) const
{
    std::map<std::string, SurfaceLoaderGen> result;
    std::regex regExpression(regex);
    for(const auto& surfaceGenerator : surfaceLoaderGenerators)
    {
        if(std::regex_match(surfaceGenerator.first, regExpression))
            result.emplace(surfaceGenerator);
    }
    return result;
}