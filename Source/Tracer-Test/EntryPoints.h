#pragma once

#include "TracerLib/TracerLogicI.h"
#include "RayLib/TracerStructs.h"

class GPUBaseAcceleratorI;

extern "C" _declspec(dllexport) TracerBaseLogicI * __stdcall GenerateBasicTracer(GPUBaseAcceleratorI& ba,
                                                                                 AcceleratorGroupList&& ag,
                                                                                 AcceleratorBatchMappings&& ab,
                                                                                 MaterialGroupList&& mg,
                                                                                 MaterialBatchMappings&& mb,
                                                                                 //
                                                                                 const TracerParameters&,
                                                                                 uint32_t,
                                                                                 const Vector2i,
                                                                                 const Vector2i,
                                                                                 const HitKey);