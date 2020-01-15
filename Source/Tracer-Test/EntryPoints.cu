#include "EntryPoints.h"
#include "TestPools.h"

extern "C" _declspec(dllexport) MaterialLogicPoolI * __stdcall GenerateTestMaterialPool()
{
    return new TestMaterialPool();
}

extern "C" _declspec(dllexport) TracerLogicPoolI * __stdcall GenerateTestTracerPool()
{
    return new TestTracerLogicPool();
}

extern "C" _declspec(dllexport) void __stdcall DeleteTestTracerPool(TracerLogicPoolI* pool)
{
    delete pool;
}

extern "C" _declspec(dllexport) void __stdcall DeleteTestMaterialPool(MaterialLogicPoolI* pool)
{
    delete pool;
}

