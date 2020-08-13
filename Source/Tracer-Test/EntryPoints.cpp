#include "EntryPoints.h"
#include "TestPools.h"

extern "C" _declspec(dllexport) TracerPoolI* __stdcall GenerateTestTracerPool()
{
    return new TestTracerPool();
}

extern "C" _declspec(dllexport) void __stdcall DeleteTestTracerPool(TracerPoolI* pool)
{
    delete pool;
}