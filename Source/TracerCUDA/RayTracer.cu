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
    , totalSamplePerPixel(0)
{}

void RayTracer::SwapAuxBuffers()
{
    std::swap(dAuxIn, dAuxOut);
}

TracerError RayTracer::Initialize()
{
    return GPUTracer::Initialize();
}

void RayTracer::UpdateFrameAnalytics(const std::string& throughputSuffix,
                                     uint32_t spp)
{
    const Vector2i& segmentSize = imgMemory.SegmentSize();
    int32_t pixCount = segmentSize[0] * segmentSize[1];
    totalSamplePerPixel += spp;

    float iterationTime = static_cast<float>(frameTimer.Elapsed<CPUTimeMillis>());
    double throughput = pixCount / iterationTime / 1'000.0;

    frameAnalytics.iterationTime = iterationTime;
    frameAnalytics.throughput = throughput;
    frameAnalytics.throughputSuffix = std::string("M ") + throughputSuffix;
    frameAnalytics.totalCPUMemoryMiB = 0;
    frameAnalytics.totalGPUMemoryMiB = 0;
    frameAnalytics.workPerPixel = totalSamplePerPixel;
    frameAnalytics.workPerPixelSuffix = " spp";
}