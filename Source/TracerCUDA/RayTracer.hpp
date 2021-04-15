#include "GenerationKernels.cuh"
#include "RayTracer.h"

template <class AuxStruct, AuxInitFunc<AuxStruct> AuxFunc>
void RayTracer::GenerateRays(const GPUCameraI* dCamera, int32_t sampleCount,
                             const AuxStruct& initialValues)
{
    int32_t sampleCountSqr = sampleCount * sampleCount;

    const Vector2i resolution = imgMemory.Resolution();
    const Vector2i pixelStart = imgMemory.SegmentOffset();
    const Vector2i pixelEnd = pixelStart + imgMemory.SegmentSize();

    Vector2i pixelCount = (pixelEnd - pixelStart);
    uint32_t totalRayCount = pixelCount[0] * pixelCount[1] * sampleCountSqr;
    size_t auxBufferSize = totalRayCount * sizeof(AuxStruct);

    // Allocate enough space for ray
    rayMemory.ResizeRayOut(totalRayCount, scene.BaseBoundaryMaterial());
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxBufferSize);

    // Basic Tracer does classic camera to light tracing
    // Thus its initial rays are from camera
    // Call multi-device
    const uint32_t TPB = StaticThreadPerBlock1D;
    // GPUSplits
    const auto splits = cudaSystem.GridStrideMultiGPUSplit(totalRayCount, TPB, 0,
                                                           KCGenerateCameraRaysGPU<AuxStruct, AuxFunc>);

    // Only use splits as guidance
    // and Split work into columns (much easier to maintain..
    // however not perfectly balanced... (as all things should be))
    int i = 0;
    Vector2i localPixelStart = Zero2i;
    for(const CudaGPU& gpu : cudaSystem.GPUList())
    {
        // If no work is splitted to this GPU skip
        if(splits[i] == 0) break;

        // Generic Args
        // Offsets
        const size_t localPixelCount1D = splits[i] / sampleCountSqr;
        const int columnCount = static_cast<int>((localPixelCount1D + pixelCount[1] - 1) / pixelCount[1]);
        const int rowCount = pixelCount[1];
        Vector2i localPixelCount = Vector2i(columnCount, rowCount);
        Vector2i localPixelEnd = Vector2i::Min(localPixelStart + localPixelCount, pixelCount);
        Vector2i localWorkCount2D = (localPixelEnd - localPixelStart) * sampleCountSqr;
        size_t localWorkCount = localWorkCount2D[0] * localWorkCount2D[1];

        // Kernel Specific Args
        // Output
        RayGMem* gRays = rayMemory.RaysOut();
        AuxStruct* gAuxiliary = static_cast<AuxStruct*>(*dAuxOut);
        // Input
        RNGGMem rngData = rngMemory.RNGData(gpu);
        ImageGMem<Vector4f> gImgData = imgMemory.GMem<Vector4f>();

        cudaSystem.SyncAllGPUs();

        // Kernel Call
        gpu.AsyncGridStrideKC_X
        (
            0, localWorkCount,
            KCGenerateCameraRaysGPU<AuxStruct, AuxFunc>,
            // Args
            // Inputs
            gRays,
            gAuxiliary,
            gImgData,
            // Input
            rngData,
            dCamera,
            sampleCount,
            resolution,
            localPixelStart,
            localPixelEnd,
            // Data to initialize auxiliary base data
            initialValues
        );

        // Adjust for next call
        localPixelStart = localPixelEnd;
        i++;
    }
    cudaSystem.SyncAllGPUs();

    SwapAuxBuffers();
    rayMemory.SwapRays();
    currentRayCount = totalRayCount;
}

template <class AuxStruct, AuxInitFunc<AuxStruct> AuxFunc>
void RayTracer::GenerateRays(const VisorCamera& cam, int32_t sampleCount,
                             const AuxStruct& initialValues)
{
    // Visor Camera is GUI represented camera
    // Its fov values are in degrees (since its more intuitive to show as such
    // on GUI) so convert it to radians
    VisorCamera camera = cam;
    camera.fov = camera.fov * MathConstants::DegToRadCoef;

    int32_t sampleCountSqr = sampleCount * sampleCount;

    const Vector2i resolution = imgMemory.Resolution();
    const Vector2i pixelStart = imgMemory.SegmentOffset();
    const Vector2i pixelEnd = pixelStart + imgMemory.SegmentSize();

    Vector2i pixelCount = (pixelEnd - pixelStart);
    uint32_t totalRayCount = pixelCount[0] * pixelCount[1] * sampleCountSqr;
    size_t auxBufferSize = totalRayCount * sizeof(AuxStruct);

    // Allocate enough space for ray
    rayMemory.ResizeRayOut(totalRayCount, scene.BaseBoundaryMaterial());
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxBufferSize);

    // Basic Ray Tracer does classic camera to light tracing
    // Thus its initial rays are from camera
    // Call multi-device
    const uint32_t TPB = StaticThreadPerBlock1D;
    // GPUSplits
    const auto splits = cudaSystem.GridStrideMultiGPUSplit(totalRayCount, TPB, 0,
                                                           KCGenerateCameraRaysCPU<AuxStruct, AuxFunc>);

    // Only use splits as guidance
    // and Split work into columns (much easier to maintain..
    // however not perfectly balanced... (as all things should be))
    int i = 0;
    Vector2i localPixelStart = Zero2i;
    for(const CudaGPU& gpu : cudaSystem.GPUList())
    {
        // If no work is splitted to this GPU skip
        if(splits[i] == 0) break;

        // Generic Args
        // Offsets
        const size_t localPixelCount1D = splits[i] / sampleCountSqr;
        const int columnCount = static_cast<int>((localPixelCount1D + pixelCount[1] - 1) / pixelCount[1]);
        const int rowCount = pixelCount[1];
        Vector2i localPixelCount = Vector2i(columnCount, rowCount);
        Vector2i localPixelEnd = Vector2i::Min(localPixelStart + localPixelCount, pixelCount);
        Vector2i localWorkCount2D = (localPixelEnd - localPixelStart) * sampleCountSqr;
        size_t localWorkCount = localWorkCount2D[0] * localWorkCount2D[1];

        // Kernel Specific Args
        // Output
        RayGMem* gRays = rayMemory.RaysOut();
        AuxStruct* gAuxiliary = static_cast<AuxStruct*>(*dAuxOut);
        // Input
        RNGGMem rngData = rngMemory.RNGData(gpu);
        ImageGMem<Vector4f> gImgData = imgMemory.GMem<Vector4f>();

        // Kernel Call
        gpu.AsyncGridStrideKC_X
        (
            0, localWorkCount,
            KCGenerateCameraRaysCPU<AuxStruct, AuxFunc>,
            // Args
            // Inputs
            gRays,
            gAuxiliary,
            gImgData,
            // Input
            rngData,
            camera,
            sampleCount,
            resolution,
            localPixelStart,
            localPixelEnd,
            // Data to initialize auxiliary base data
            initialValues
        );
        // Adjust for next call
        localPixelStart = localPixelEnd;
        i++;
    }
    cudaSystem.SyncAllGPUs();

    SwapAuxBuffers();
    rayMemory.SwapRays();
    currentRayCount = totalRayCount;
}