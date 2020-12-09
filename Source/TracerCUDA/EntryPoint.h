#pragma once

#include "RayLib/TracerSystemI.h"

extern "C" _declspec(dllexport)
TracerSystemI* __stdcall GenerateTracerSystem();

extern "C" _declspec(dllexport)
void __stdcall DeleteTracerSystem(TracerSystemI*);