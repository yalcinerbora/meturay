#include "EntryPoints.h"
#include "BasicTracerPools.h"

extern "C" _declspec(dllexport) TracerPoolI* __stdcall GenerateBasicTracerPool()
{
    return new BasicTracerPool();
}

extern "C" _declspec(dllexport) void __stdcall DeleteBasicTracerPool(TracerPoolI* pool)
{
    delete pool;
}