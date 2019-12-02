#include "EntryPoint.h"
#include "AssimpSurfaceLoaderPool.h"

extern "C" _declspec(dllexport) SurfaceLoaderPoolI * __stdcall GenerateAssimpPool()
{
    return new AssimpSurfaceLoaderPool();
}

extern "C" _declspec(dllexport) void __stdcall DeleteAssimpPool(SurfaceLoaderPoolI * sPool)
{
    return delete sPool;
}