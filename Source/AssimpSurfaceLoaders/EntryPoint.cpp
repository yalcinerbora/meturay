#include "EntryPoint.h"
#include "AssimpSurfaceLoaderPool.h"

extern "C" METU_SHARED_ASSIMP_ENTRY_POINT
SurfaceLoaderPoolI* GenerateAssimpPool()
{
    return new AssimpSurfaceLoaderPool();
}

extern "C" METU_SHARED_ASSIMP_ENTRY_POINT 
void DeleteAssimpPool(SurfaceLoaderPoolI * sPool)
{
    return delete sPool;
}