#pragma once

#include "TracerLib/TracerLogicI.h"
#include "RayLib/TracerStructs.h"

#include "TracerLib/TracerLogicPools.h"

extern "C" _declspec(dllexport) MaterialLogicPoolI* __stdcall GenerateTestMaterialPool();

extern "C" _declspec(dllexport) TracerLogicPoolI * __stdcall GenerateTestTracerPool();

extern "C" _declspec(dllexport) void __stdcall DeleteTestTracerPool(TracerLogicPoolI*);

extern "C" _declspec(dllexport) void __stdcall DeleteTestMaterialPool(MaterialLogicPoolI*);