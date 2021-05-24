#include "EntryPoint.h"
#include "TracerSystemCUDA.h"

extern "C" METU_SHARED_TRACER_CUDA_ENTRY_POINT
TracerSystemI* __stdcall GenerateTracerSystem()
{
    return new TracerSystemCUDA();
}

extern "C" METU_SHARED_TRACER_CUDA_ENTRY_POINT
void __stdcall DeleteTracerSystem(TracerSystemI* ts)
{
    if(ts) delete ts;
}