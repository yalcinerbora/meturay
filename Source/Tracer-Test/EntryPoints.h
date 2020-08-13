#pragma once

#include "RayLib/TracerStructs.h"

#include "TracerLib/TracerLogicPools.h"

extern "C" _declspec(dllexport) TracerPoolI * __stdcall GenerateTestTracerPool();

extern "C" _declspec(dllexport) void __stdcall DeleteTestTracerPool(TracerPoolI*);