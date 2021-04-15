#include "RayTracer.h"

#include "RayLib/TracerError.h"
#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"

#include "RayMemory.h"
#include "ImageMemory.h"
#include "GenerationKernels.cuh"

RayTracer::RayTracer(const CudaSystem& s,
                     const GPUSceneI& scene,
                     const TracerParameters& param)
    : GPUTracer(s, scene, param)
    , scene(scene)
    , dAuxIn(&auxBuffer0)
    , dAuxOut(&auxBuffer1)
{}

void RayTracer::SwapAuxBuffers()
{
    std::swap(dAuxIn, dAuxOut);
}

TracerError RayTracer::Initialize()
{
    return GPUTracer::Initialize();
}