#include "EntryPoint.h"
#include "TracerSystemCUDA.h"

extern "C" METU_SHARED_TRACER_CUDA_ENTRY_POINT
TracerSystemI* GenerateTracerSystem()
{
    return new TracerSystemCUDA();
}

extern "C" METU_SHARED_TRACER_CUDA_ENTRY_POINT
void DeleteTracerSystem(TracerSystemI* ts)
{
    if(ts) delete ts;
}