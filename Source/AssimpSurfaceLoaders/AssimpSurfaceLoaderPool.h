#pragma once

#include "RayLib/SurfaceLoaderPoolI.h"
#include <assimp/Importer.hpp>

using AssimpSurfaceLoaderGeneratorFunc = SurfaceLoaderI* (*)(Assimp::Importer&,
                                                             const std::string & scenePath,
                                                             const SceneNodeI&,
                                                             double time);

class AssimpSurfaceLoaderGen : public SurfaceLoaderGenI
{
    private:
        Assimp::Importer&                                   importer;
        AssimpSurfaceLoaderGeneratorFunc                    gFunc;
        ObjDestroyerFunc<SurfaceLoaderI>                    dFunc;

    public:
        // Constructor & Destructor
        AssimpSurfaceLoaderGen(Assimp::Importer& i,
                               AssimpSurfaceLoaderGeneratorFunc g,
                               ObjDestroyerFunc<SurfaceLoaderI> d)
            : importer(i)
            , gFunc(g)
            , dFunc(d)
        {}

        SurfaceLoaderPtr operator()(const std::string& scenePath,
                                    const SceneNodeI& n,
                                    double time) const override
        {
            SurfaceLoaderI* loader = gFunc(importer, scenePath, n, time);
            return SurfaceLoaderPtr(loader, dFunc);
        }
};

class AssimpSurfaceLoaderPool : public SurfaceLoaderPoolI
{
    private:
        // TODO: Do a multi thread system(for this)
        Assimp::Importer            importer;

        //
        AssimpSurfaceLoaderGen      objSurfaceLoader;

    protected:
    public:
        // Constructors & Destructor
                                    AssimpSurfaceLoaderPool();
                                    AssimpSurfaceLoaderPool(const AssimpSurfaceLoaderPool&) = delete;
        AssimpSurfaceLoaderPool&    operator=(const AssimpSurfaceLoaderPool&) = delete;
                                    ~AssimpSurfaceLoaderPool();
};