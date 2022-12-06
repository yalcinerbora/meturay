#include "RayTracer.h"

#include "RayLib/TracerError.h"
#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"

#include "RayMemory.h"
#include "ImageMemory.h"
#include "GenerationKernels.cuh"
#include "GPUReconFilterI.h"
#include "TracerDebug.h"

RayTracer::RayTracer(const CudaSystem& s,
                     const GPUSceneI& scene,
                     const TracerParameters& param)
    : GPUTracer(s, scene, param)
    , dAuxIn(&auxBuffer0)
    , dAuxOut(&auxBuffer1)
    , scene(scene)
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

void RayTracer::ResetSPP()
{
    totalSamplePerPixel = 0;
}

void RayTracer::UpdateFrameAnalytics(const std::string& throughputSuffix,
                                     uint32_t spp)
{
    const Vector2i& segmentSize = imgMemory.SegmentSize();
    int32_t pixCount = segmentSize[0] * segmentSize[1];
    totalSamplePerPixel += spp;

    float iterationTime = static_cast<float>(frameTimer.Elapsed<CPUTimeMillis>());
    double throughput = pixCount * spp / iterationTime / 1'000.0;

    double totalMemMiB = static_cast<double>(cudaSystem.TotalMemory()) / 1024.0 / 1024.0;
    double usedMemMiB = static_cast<double>(TotalGPUMemoryUsed()) / 1024.0 / 1024.0;

    frameAnalytics.iterationTime = iterationTime;
    frameAnalytics.throughput = throughput;
    frameAnalytics.throughputSuffix = std::string("M ") + throughputSuffix;
    frameAnalytics.usedGPUMemoryMiB = usedMemMiB;
    frameAnalytics.totalGPUMemoryMiB = totalMemMiB;
    frameAnalytics.workPerPixel = totalSamplePerPixel;
    frameAnalytics.workPerPixelSuffix = " spp";
}

size_t RayTracer::TotalGPUMemoryUsed() const
{
    size_t reconFilterMemSize = (reconFilter) ? reconFilter->UsedGPUMemory() : 0;

    return (GPUTracer::TotalGPUMemoryUsed() +
            sampleMemory.UsedGPUMemory() +
            reconFilterMemSize +
            auxBuffer0.Size() +
            auxBuffer1.Size());
}

void RayTracer::AttachReconFilter(GPUReconFilterI* f)
{
    reconFilter = f;
}

void RayTracer::ResetImage()
{
    ResetSPP();
    GPUTracer::ResetImage();
}

void RayTracer::Finalize()
{
    if(crashed) return;
    const auto sampleGMem = std::as_const(sampleMemory).GMem<Vector4f>();

    if(reconFilter != nullptr)
        reconFilter->FilterToImg(imgMemory,
                                 sampleGMem.gValues,
                                 sampleGMem.gImgCoords,
                                 sampleMemory.SampleCount(),
                                 cudaSystem);
    else
    {
        METU_ERROR_LOG("Reconstruction Filter did not set!");
        crashed = true;
    }
    GPUTracer::Finalize();
}