#include "EntryPoint.h"
#include "TracerSystemCUDA.h"

extern "C" _declspec(dllexport)
TracerSystemI* __stdcall GenerateTracerSystem()
{
    return new TracerSystemCUDA();
}

extern "C" _declspec(dllexport)
void __stdcall DeleteTracerSystem(TracerSystemI* ts)
{
    if(ts) delete ts;
}