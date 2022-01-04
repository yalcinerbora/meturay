#pragma once

/**

Camera Ray Generation Kernel

Uses stratified sampling

*/

#include <cstdint>
#include <cuda_runtime.h>

#include "RayLib/Vector.h"

#include "RayStructs.h"
#include "ImageStructs.h"
#include "RNGenerator.h"
#include "GPUCameraI.h"
#include "CameraFunctions.h"

#include "ImageFunctions.cuh"
#include "CudaSystem.hpp"

// Templated Camera Ray Generation Kernel
template<class RayAuxData, class AuxInitFunctor,
         class RNG>
__device__ __forceinline__
void GenerateCameraRaysGPU(// Output
                           RayGMem* gRays,
                           RayAuxData* gAuxiliary,
                           ImageGMem<Vector4f> imgMem,
                           // I-O
                           RNGeneratorGPUI** gRNGs,
                           // Input
                           const GPUCameraI& gCamera,
                           const uint32_t samplePerLocation,
                           const Vector2i resolution,
                           const Vector2i pixelStart,
                           const Vector2i pixelCount,
                           // Functor to initialize auxiliary base data
                           const AuxInitFunctor auxInitF,
                           // Options
                           bool incSampleCount,
                           bool antiAliasOn)
{
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);

    // Total work
    const Vector2i totalSamples = Vector2i(pixelCount[0] * samplePerLocation,
                                           pixelCount[1] * samplePerLocation);
    const uint32_t totalWorkCount = pixelCount[0] * samplePerLocation *
                                    pixelCount[1] * samplePerLocation;

    // Kernel Grid-Stride Loop
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalWorkCount;
        threadId += (blockDim.x * gridDim.x))
    {
        Vector2i threadId2d = Vector2i(threadId % (pixelCount[0] * samplePerLocation),
                                       threadId / (pixelCount[0] * samplePerLocation));
        Vector2i globalSampleId = (pixelStart * samplePerLocation) + threadId2d;
        Vector2i globalPixelId = pixelStart + (threadId2d / samplePerLocation);

        RayReg ray;
        gCamera.GenerateRay(ray,
                            //
                            globalSampleId,
                            totalSamples,
                            rng,
                            //
                            antiAliasOn);
        // Generate Required Parameters
        Vector2i pixelSampleId = threadId2d % samplePerLocation;
        Vector2i localPixelId = globalPixelId - pixelStart;
        uint32_t pixelIdLinear = localPixelId[1] * pixelCount[0] + localPixelId[0];
        uint32_t sampleIdLinear = pixelSampleId[1] * samplePerLocation + pixelSampleId[0];

        // Write Ray
        ray.Update(gRays, threadId);

        // Write Auxiliary Data
        auxInitF(gAuxiliary[threadId],
                 // Input
                 ray,
                 // Index
                 gCamera.MediumIndex(),
                 pixelIdLinear,
                 sampleIdLinear);

        // Initialize Samples
        if(incSampleCount) ImageAddSample(imgMem, pixelIdLinear, 1);
    }
}

// Templated Camera Ray Generation Kernel
template<class RayAuxData, class AuxInitFunctor, class RNG>
__global__ CUDA_LAUNCH_BOUNDS_1D
void KCGenCameraRaysFromArrayGPU(// Output
                                 RayGMem* gRays,
                                 RayAuxData* gAuxiliary,
                                 ImageGMem<Vector4f> imgMem,
                                 // I-O
                                 RNGeneratorGPUI** gRNGs,
                                 // Input
                                 const GPUCameraI** gCameras,
                                 const uint32_t sceneCamId,
                                 const uint32_t samplePerLocation,
                                 const Vector2i resolution,
                                 const Vector2i pixelStart,
                                 const Vector2i pixelCount,
                                 // Functor to initialize auxiliary base data
                                 const AuxInitFunctor auxInitF,
                                 // Options
                                 bool incSampleCount,
                                 bool antiAliasOn)
{
    // Fetch Camera
    const GPUCameraI* gCam = gCameras[sceneCamId];

    GenerateCameraRaysGPU<RayAuxData, AuxInitFunctor, RNG>
    (
        // Output
        gRays,
        gAuxiliary,
        imgMem,
        // Input
        gRNGs,
        *gCam,
        samplePerLocation,
        resolution,
        pixelStart,
        pixelCount,
        // Functor to initialize auxiliary base data
        auxInitF,
        // Options
        incSampleCount,
        antiAliasOn
    );
}

// Templated Camera Ray Generation Kernel
template<class RayAuxData, class AuxInitFunctor,
         class RNG>
__global__ CUDA_LAUNCH_BOUNDS_1D
void KCGenCameraRaysFromObjectGPU(// Output
                                  RayGMem* gRays,
                                  RayAuxData* gAuxiliary,
                                  ImageGMem<Vector4f> imgMem,
                                  // I-O
                                  RNGeneratorGPUI** gRNGs,
                                  // Input
                                  const GPUCameraI& gCamera,
                                  const uint32_t samplePerLocation,
                                  const Vector2i resolution,
                                  const Vector2i pixelStart,
                                  const Vector2i pixelCount,
                                  // Functor to initialize auxiliary base data
                                  const AuxInitFunctor auxInitF,
                                  // Options
                                  bool incSampleCount,
                                  bool antiAliasOn)
{
    GenerateCameraRaysGPU<RayAuxData, AuxInitFunctor, RNG>
    (
        // Output
        gRays,
        gAuxiliary,
        imgMem,
        // Input
        gRNGs,
        gCamera,
        samplePerLocation,
        resolution,
        pixelStart,
        pixelCount,
        // Functor to initialize auxiliary base data
        auxInitF,
        // Options
        incSampleCount,
        antiAliasOn
    );
}