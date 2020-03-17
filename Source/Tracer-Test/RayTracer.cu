#include "RayTracer.h"

#include "RayLib/TracerError.h"
#include "RayLib/GPUSceneI.h"
#include "RayLib/TracerCallbacksI.h"

#include "TracerLib/RayMemory.h"
#include "TracerLib/ImageMemory.h"
#include "TracerLib/CudaConstants.h"
#include "TracerLib/GenerationKernels.cuh"
#include "TracerLib/LightCameraKernels.h"

#include "RayAuxStruct.h"

//std::ostream& operator<<(std::ostream& stream, const RayAuxBasic& v)
//{
//    stream << std::setw(0)
//            << v.pixelId << ", "
//            << v.pixelSampleId << ", "
//            << "{" << v.totalRadiance[0]
//            << "," << v.totalRadiance[0]
//            << "," << v.totalRadiance[0] << "}";
//    return stream;
//}

__device__ __host__
inline void RayInitBasic(RayAuxBasic* gOutBasic,
                         const uint32_t writeLoc,
                         // Input
                         const RayAuxBasic& defaults,
                         const RayReg& ray,
                         // Index
                         const uint32_t localPixelId,
                         const uint32_t pixelSampleId)
{
    RayAuxBasic init = defaults;
    init.pixelId = localPixelId;
    gOutBasic[writeLoc] = init;
}

RayTracer::RayTracer(const CudaSystem& s, 
                     const GPUSceneI& scene,
                     const TracerParameters& param)
    : GPUTracer(s, scene, param)
    , scene(scene)
    , dCameraPtr(nullptr)
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

void RayTracer::GenerateRays(const GPUCameraI* dCamera, int32_t sampleCount)
{
    int32_t sampleCountSqr = sampleCount * sampleCount;

    const Vector2i resolution = imgMemory.Resolution();
    const Vector2i pixelStart = imgMemory.SegmentOffset();
    const Vector2i pixelEnd = pixelStart + imgMemory.SegmentSize();

    Vector2i pixelCount = (pixelEnd - pixelStart);
    uint32_t totalRayCount = pixelCount[0] * pixelCount[1] * sampleCountSqr;
    size_t auxBufferSize = totalRayCount + sizeof(RayAuxBasic);

    // Allocate enough space for ray
    rayMemory.ResizeRayOut(totalRayCount, scene.BaseBoundaryMaterial());
    DeviceMemory::EnlargeBuffer(*dAuxOut, auxBufferSize);

    // Basic Tracer does classic camera to light tracing
    // Thus its initial rays are from camera
    // Call multi-device
    const uint32_t TPB = StaticThreadPerBlock1D;
    // GPUSplits
    const auto splits = cudaSystem.GridStrideMultiGPUSplit(totalRayCount, TPB, 0,
                                                           KCGenerateCameraRaysGPU<RayAuxBasic, RayInitBasic>);

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
        RayAuxBasic* gAuxiliary = static_cast<RayAuxBasic*>(*dAuxOut);
        // Input
        RNGGMem rngData = rngMemory.RNGData(gpu);
        ImageGMem<Vector4f> gImgData = imgMemory.GMem<Vector4f>();

        // Kernel Call
        gpu.AsyncGridStrideKC_X
        (
            0, localWorkCount,
            KCGenerateCameraRaysGPU<RayAuxBasic, RayInitBasic>,
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
            InitialBasicAux
        );

        // Adjust for next call
        localPixelStart = localPixelEnd;
        i++;
    }

    SwapAuxBuffers();
    rayMemory.SwapRays();
    currentRayCount = totalRayCount;
}

void RayTracer::LoadCameraToGPU(const CPUCamera& c)
{
    // Load it to the tmep buffer
    DeviceMemory::EnlargeBuffer(tempCameraBuffer,
                                LightCameraKernels::CameraClassesUnionSize() + 
                                sizeof(GPUCameraI*));
    Byte* memPtr = static_cast<Byte*>(tempCameraBuffer);
    dCameraPtr = reinterpret_cast<GPUCameraI*>(memPtr + LightCameraKernels::CameraClassesUnionSize());

    LightCameraKernels::ConstructSingleCamera(dCameraPtr,
                                              memPtr, c,
                                              cudaSystem);
    // Generate Rays over this camera
}