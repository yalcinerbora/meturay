#pragma once

#include "RayLib/TracerStructs.h"

#include "TracerLib/TracerLogicPools.h"

extern "C" _declspec(dllexport) MaterialLogicPoolI* __stdcall GenerateTestMaterialPool();

extern "C" _declspec(dllexport) TracerPoolI * __stdcall GenerateTestTracerPool();

extern "C" _declspec(dllexport) void __stdcall DeleteTestTracerPool(TracerPoolI*);

extern "C" _declspec(dllexport) void __stdcall DeleteTestMaterialPool(MaterialLogicPoolI*);