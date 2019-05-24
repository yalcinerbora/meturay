#include "TracerLogics.cuh"

#include "RayLib/TracerError.h"
#include "TracerLib/GPUScene.h"
#include "TracerLib/RayMemory.h"

TracerBasic::TracerBasic(GPUBaseAcceleratorI& ba,
                         AcceleratorGroupList&& ag,
                         AcceleratorBatchMappings&& ab,
                         MaterialGroupList&& mg,
                         MaterialBatchMappings&& mb,
                         //
                         const TracerParameters& options,
                         uint32_t hitStructSize,
                         const Vector2i maxMats,
                         const Vector2i maxAccels,
                         const HitKey baseBoundMatKey)
    : TracerBaseLogic(ba,
                      std::move(ag), std::move(ab),
                      std::move(mg), std::move(mb),
                      options,
                      initals,
                      hitStructSize,
                      maxMats,
                      maxAccels,
                      baseBoundMatKey)
{}

TracerError TracerBasic::Initialize()
{
    return TracerError::OK;
}

uint32_t TracerBasic::GenerateRays(RayMemory& rayMem, RNGMemory& rngMem,
                                   const GPUScene& scene,
                                   const CameraPerspective& cam,
                                   int samplePerLocation,
                                   Vector2i resolution,
                                   Vector2i pixelStart,
                                   Vector2i pixelEnd)
{
    pixelEnd = Vector2i::Min(resolution, pixelEnd);
    Vector2i pixelCount = (pixelEnd - pixelStart);
    uint32_t totalRayCount = pixelCount[0] * samplePerLocation *
                             pixelCount[1] * samplePerLocation;

    // Allocate enough space for ray
    rayMem.ResizeRayOut(totalRayCount, PerRayAuxDataSize(),
                        SceneBaseBoundMatKey());

    // Basic Tracer does classic camera to light tracing
    // Thus its initial rays are from camera

    // Call multi-device
    const uint32_t TPB = StaticThreadPerBlock1D;
    const uint32_t shMemSize = rngMem.SharedMemorySize(TPB);

    // GPUSplits
    const auto splits = CudaSystem::GridStrideMultiGPUSplit(totalRayCount, TPB, shMemSize,
                                                            KCGenerateCameraRays<RayAuxData, AuxFunc>);


    // Only use splits as guidance
    // and Split work into columns (much easier to maintain..
    // however not perfectly balanced... (as all things should be))
    Vector2i localPixelStart = Zero2i;
    for(int i = 0; i < static_cast<int>(CudaSystem::GPUList().size()); i++)
    {
        // If no work is splitted to this GPU skip
        if(splits[i] == 0) break;

        // Arguments
        const CudaGPU& gpu = CudaSystem::GPUList()[i];
        const int gpuId = gpu.DeviceId();
        // Generic Args
        // Offsets
        const size_t localPixelCount1D = splits[i] / samplePerLocation / samplePerLocation;
        const int columnCount = static_cast<int>((localPixelCount1D + pixelCount[1] - 1) / pixelCount[1]);
        const int rowCount = pixelCount[1];
        Vector2i localPixelCount = Vector2i(columnCount, rowCount);
        Vector2i localPixelEnd = Vector2i::Min(localPixelStart + localPixelCount, pixelCount);
        Vector2i localWorkCount2D = (localPixelEnd - localPixelStart) * samplePerLocation * samplePerLocation;
        size_t localWorkCount = localWorkCount2D[0] * localWorkCount2D[1];

        // Kernel Specific Args
        // Output
        RayGMem* gRays = rayMem.RaysOut();
        RayAuxData* gAuxiliary = rayMem.RayAuxOut<RayAuxData>();
        // Input
        RNGGMem rngData = rngMem.RNGData(gpuId);

        // Kernel Call
        CudaSystem::AsyncGridStrideKC_X
        (
            gpuId, shMemSize, localWorkCount,
            KCGenerateCameraRays<RayAuxData, AuxFunc>,
            // Args
            // Inputs
            gRays,
            gAuxiliary,
            // Input
            rngData,
            cam,
            samplePerLocation,
            resolution,
            localPixelStart,
            localPixelEnd,
            // Data to initialize auxiliary base data
            initialValues
        );

        // Adjust for next call
        localPixelStart = localPixelEnd;
    }
    return totalRayCount;
}