#pragma once

#include "RayLib/SurfaceLoaderPoolI.h"
#include <assimp/Importer.hpp>

class AssimpSurfaceLoaderPool : public SurfaceLoaderPoolI
{
    private:
        // TODO: Do a multi thread system(for this)
        Assimp::Importer            importer;

    protected:
    public:
        // Constructors & Destructor
                                    AssimpSurfaceLoaderPool();
                                    AssimpSurfaceLoaderPool(const AssimpSurfaceLoaderPool&) = delete;
        AssimpSurfaceLoaderPool&    operator=(const AssimpSurfaceLoaderPool&) = delete;
                                    ~AssimpSurfaceLoaderPool();
};