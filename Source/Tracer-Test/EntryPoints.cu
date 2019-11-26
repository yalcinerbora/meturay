#include "EntryPoints.h"
#include "TracerLogics.cuh"

extern "C" _declspec(dllexport) TracerBaseLogicI* __stdcall GenerateBasicTracer(GPUBaseAcceleratorI& ba,
                                                                                AcceleratorGroupList&& ag,
                                                                                AcceleratorBatchMappings&& ab,
                                                                                MaterialGroupList&& mg,
                                                                                MaterialBatchMappings&& mb,
                                                                                //
                                                                                const TracerParameters& params,
                                                                                uint32_t hitStructSize,
                                                                                const Vector2i maxMats,
                                                                                const Vector2i maxAccels,
                                                                                const HitKey baseBoundMatKey)
{
    return new TracerBasic(ba, 
                           std::move(ag), std::move(ab), 
                           std::move(mg), std::move(mb), 
                           params,
                           hitStructSize,
                           maxMats, maxAccels,
                           baseBoundMatKey);
}

extern "C" _declspec(dllexport) void __stdcall DeleteBasicTracer(TracerBaseLogicI * tGen)
{
    return delete tGen;
}