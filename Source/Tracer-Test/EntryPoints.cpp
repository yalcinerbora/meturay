#include "EntryPoints.h"
#include "TracerGenerators.h"

extern "C" _declspec(dllexport) TracerLogicGeneratorI* __stdcall GenerateBasicTracer()
{
    return new BasicTracerLogicGenerator();
}

extern "C" _declspec(dllexport) void __stdcall DeleteBasicTracer(TracerLogicGeneratorI* tGen)
{
    return delete tGen;
}