#pragma once

#include "RayLib/System.h"

#ifdef METU_SHARED_GFG_LOADER
#define METU_SHARED_GFG_ENTRY_POINT MRAY_DLL_EXPORT
#else
#define METU_SHARED_GFG_ENTRY_POINT MRAY_DLL_IMPORT
#endif

class SurfaceLoaderPoolI;

extern "C" METU_SHARED_GFG_ENTRY_POINT
SurfaceLoaderPoolI* GenerateGFGPool();

extern "C" METU_SHARED_GFG_ENTRY_POINT
void DeleteGFGPool(SurfaceLoaderPoolI* tGen);