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
         class RNG, class T>
__device__ inline
void GenerateCameraRaysGPU(// Output
                           RayGMem* gRays,
                           RayAuxData* gAuxiliary,
                           CamSampleGMem<T>& sampleMem,
                           // I-O
                           RNGeneratorGPUI** gRNGs,
                           // Input
                           const GPUCameraI& gCamera,
                           const uint32_t samplesIssuedSoFar,
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
        Vector2f cameraWindowSpaceCoords;
        gCamera.GenerateRay(ray,
                            cameraWindowSpaceCoords,
                            //
                            globalSampleId,
                            totalSamples,
                            rng,
                            //
                            antiAliasOn);
        // "Generate Ray" function returns normalized coordinates
        // convert it to image space (0, resolution)
        Vector2 imgSpaceCoords = cameraWindowSpaceCoords * Vector2f(resolution);

        // Generate Required Parameters
        Vector2i pixelSampleId = threadId2d % samplePerLocation;
        Vector2i localPixelId = globalPixelId - pixelStart;
        //uint32_t pixelIdLinear = localPixelId[1] * pixelCount[0] + localPixelId[0];
        uint32_t localSampleIdLinear = pixelSampleId[1] * samplePerLocation + pixelSampleId[0];

        // Write Sample
        uint32_t globalSampleId1D = samplesIssuedSoFar + threadId;
        sampleMem.gValues[globalSampleId1D] = T(0);
        sampleMem.gImgCoords[globalSampleId1D] = imgSpaceCoords;

        //printf("W[%u] camCoord (%f, %f), imgCoord (%f, %f)\n",
        //       globalSampleId1D,
        //       cameraWindowSpaceCoords[0],
        //       cameraWindowSpaceCoords[1],
        //       imgSpaceCoords[0], imgSpaceCoords[1]);

        // Write Ray
        ray.Update(gRays, threadId);

        // Write Auxiliary Data
        auxInitF(gAuxiliary[threadId],
                 // Input
                 ray,
                 // Index
                 gCamera.MediumIndex(),
                 globalSampleId1D,
                 localSampleIdLinear);
    }
}

// Templated Camera Ray Generation Kernel
template<class RayAuxData, class AuxInitFunctor, class RNG, class T>
__global__ CUDA_LAUNCH_BOUNDS_1D
void KCGenCameraRaysFromArrayGPU(// Output
                                 RayGMem* gRays,
                                 RayAuxData* gAuxiliary,
                                 CamSampleGMem<T> sampleMem,
                                 // I-O
                                 RNGeneratorGPUI** gRNGs,
                                 // Input
                                 const GPUCameraI** gCameras,
                                 const uint32_t sceneCamId,
                                 const uint32_t samplesIssuedSoFar,
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

    GenerateCameraRaysGPU<RayAuxData, AuxInitFunctor, RNG, T>
    (
        // Output
        gRays,
        gAuxiliary,
        sampleMem,
        // Input
        gRNGs,
        *gCam,
        samplesIssuedSoFar,
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
         class RNG, class T>
__global__ CUDA_LAUNCH_BOUNDS_1D
void KCGenCameraRaysFromObjectGPU(// Output
                                  RayGMem* gRays,
                                  RayAuxData* gAuxiliary,
                                  CamSampleGMem<T> sampleMem,
                                  // I-O
                                  RNGeneratorGPUI** gRNGs,
                                  // Input
                                  const GPUCameraI& gCamera,
                                  const uint32_t samplesIssuedSoFar,
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
    GenerateCameraRaysGPU<RayAuxData, AuxInitFunctor, RNG, T>
    (
        // Output
        gRays,
        gAuxiliary,
        sampleMem,
        // Input
        gRNGs,
        gCamera,
        samplesIssuedSoFar,
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