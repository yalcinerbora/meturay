#pragma once

#include "TracerLib/TracerLogicI.h"
#include "RayLib/TracerStructs.h"

#include "TracerLib/TracerLogicPools.h"

class GPUBaseAcceleratorI;

extern "C" _declspec(dllexport) TracerBaseLogicI* __stdcall GenerateBasicTracer(GPUBaseAcceleratorI& ba,
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

extern "C" _declspec(dllexport) MaterialLogicPoolI* __stdcall GenerateTestMaterialPool();

extern "C" _declspec(dllexport) void __stdcall DeleteBasicTracer(TracerBaseLogicI* tGen);

extern "C" _declspec(dllexport) void __stdcall DeleteTestMaterialPool(MaterialLogicPoolI* pool);