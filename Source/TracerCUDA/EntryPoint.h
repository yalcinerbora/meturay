#pragma once

#include "RayLib/TracerSystemI.h"
#include "RayLib/System.h"

#ifdef METU_SHARED_TRACER_CUDA
#define METU_SHARED_TRACER_CUDA_ENTRY_POINT MRAY_DLL_EXPORT
#else
#define METU_SHARED_TRACER_CUDA_ENTRY_POINT MRAY_DLL_IMPORT
#endif

extern "C" METU_SHARED_TRACER_CUDA_ENTRY_POINT
TracerSystemI* GenerateTracerSystem();

extern "C" METU_SHARED_TRACER_CUDA_ENTRY_POINT
void DeleteTracerSystem(TracerSystemI*);