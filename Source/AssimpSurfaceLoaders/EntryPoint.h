#pragma once

#include "RayLib/System.h"

#ifdef METU_SHARED_ASSIMP_LOADER
#define METU_SHARED_ASSIMP_ENTRY_POINT MRAY_DLL_EXPORT
#else
#define METU_SHARED_ASSIMP_ENTRY_POINT MRAY_DLL_IMPORT
#endif

class SurfaceLoaderPoolI;

extern "C" METU_SHARED_ASSIMP_ENTRY_POINT 
SurfaceLoaderPoolI* GenerateAssimpPool();

extern "C" METU_SHARED_ASSIMP_ENTRY_POINT 
void DeleteAssimpPool(SurfaceLoaderPoolI * tGen);