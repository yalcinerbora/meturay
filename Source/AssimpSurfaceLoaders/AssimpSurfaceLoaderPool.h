#pragma once

#include "RayLib/SurfaceLoaderPoolI.h"
#include <assimp/Importer.hpp>
#include <list>

using AssimpSurfaceLoaderGeneratorFunc = SurfaceLoaderI* (*)(Assimp::Importer&,
                                                             const std::string& scenePath,
                                                             const std::string& fileExt,
                                                             const SceneNodeI&,
                                                             double time);

class AssimpSurfaceLoaderGen : public SurfaceLoaderGenI
{
    private:
        const std::string                                   fileExt;
        Assimp::Importer&                                   importer;
        AssimpSurfaceLoaderGeneratorFunc                    gFunc;
        ObjDestroyerFunc<SurfaceLoaderI>                    dFunc;

    public:
        // Constructor & Destructor
        AssimpSurfaceLoaderGen(Assimp::Importer& i,
                               const std::string& fileExt,
                               AssimpSurfaceLoaderGeneratorFunc g,
                               ObjDestroyerFunc<SurfaceLoaderI> d)
            : fileExt(fileExt)
            , importer(i)
            , gFunc(g)
            , dFunc(d)
        {}

        SurfaceLoaderPtr operator()(const std::string& scenePath,
                                    const SceneNodeI& n,
                                    double time) const override
        {
            SurfaceLoaderI* loader = gFunc(importer, scenePath, fileExt, n, time);
            return SurfaceLoaderPtr(loader, dFunc);
        }
};

class AssimpSurfaceLoaderPool : public SurfaceLoaderPoolI
{
    private:
        static constexpr std::string_view    AssimpPrefix = "assimp_";
        static constexpr std::string_view    AssimpLogFileName = "assimp_log";

        // TODO: Do a multi thread system(for this)
        Assimp::Importer                            importer;

        // Single Meta Surface loader for known file types
        std::list<AssimpSurfaceLoaderGen>          assimpSLGenerators;

    protected:
    public:
        // Constructors & Destructor
                                    AssimpSurfaceLoaderPool();
                                    AssimpSurfaceLoaderPool(const AssimpSurfaceLoaderPool&) = delete;
        AssimpSurfaceLoaderPool&    operator=(const AssimpSurfaceLoaderPool&) = delete;
                                    ~AssimpSurfaceLoaderPool();
};