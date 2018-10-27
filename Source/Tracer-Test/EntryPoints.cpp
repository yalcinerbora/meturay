#include "EntryPoints.h"
#include "TracerGenerators.h"

extern "C" _declspec(dllexport) TracerLogicGeneratorI* __stdcall GenBasicTracer()
{
	return new BasicTracerLogicGenerator();
}

extern "C" _declspec(dllexport) void __stdcall DestBasicTracer(TracerLogicGeneratorI* tGen)
{
	return delete tGen;
}