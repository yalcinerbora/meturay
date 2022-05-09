#include "EntryPoint.h"
#include "GFGSurfaceLoaderPool.h"

extern "C" METU_SHARED_GFG_ENTRY_POINT
SurfaceLoaderPoolI* GenerateGFGPool()
{
    return new GFGSurfaceLoaderPool();
}

extern "C" METU_SHARED_GFG_ENTRY_POINT
void DeleteGFGPool(SurfaceLoaderPoolI * sPool)
{
    return delete sPool;
}