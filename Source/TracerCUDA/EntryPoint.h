#pragma once

#include "RayLib/TracerSystemI.h"

#ifdef METU_SHARED_TRACER_CUDA
#define METU_SHARED_TRACER_CUDA_ENTRY_POINT __declspec(dllexport)
#else
#define METU_SHARED_TRACER_CUDA_ENTRY_POINT __declspec(dllimport)
#endif

extern "C" METU_SHARED_TRACER_CUDA_ENTRY_POINT
TracerSystemI* __stdcall GenerateTracerSystem();

extern "C" METU_SHARED_TRACER_CUDA_ENTRY_POINT
void __stdcall DeleteTracerSystem(TracerSystemI*);