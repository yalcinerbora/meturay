#pragma once

/**

Camera Ray Generation Kernel

Uses statified sampling

*/

#include <cstdint>
#include <cuda_runtime.h>

#include "RayLib/Vector.h"
#include "RayLib/VisorCamera.h"

#include "RayStructs.h"
#include "ImageStructs.h"
#include "Random.cuh"
#include "GPUCameraI.h"
#include "CameraFunctions.h"

#include "ImageFunctions.cuh"
#include "CudaSystem.hpp"

// Templated Camera Ray Generation Kernel
template<class RayAuxData, class AuxInitFunctor>
__global__ CUDA_LAUNCH_BOUNDS_1D
void KCGenerateCameraRaysCPU(// Output
                             RayGMem* gRays,
                             RayAuxData* gAuxiliary,
                             ImageGMem<Vector4f> imgMem,
                             // Input
                             RNGGMem gRNGStates,
                             const VisorCamera cam,
                             const uint32_t samplePerLocation,
                             const Vector2i resolution,
                             const Vector2i pixelStart,
                             const Vector2i pixelCount,
                             // Functor to initialize auxiliary base data
                             const AuxInitFunctor auxInitF,
                             // Options
                             bool antiAliasOn)
{
    RandomGPU rng(gRNGStates, LINEAR_GLOBAL_ID);

    // Total work
    const uint32_t totalWorkCount = pixelCount[0] * samplePerLocation *
                                    pixelCount[1] * samplePerLocation;

    // Find world space window sizes
    float widthHalf = tanf(cam.fov[0] * 0.5f) * cam.nearPlane;
    float heightHalf = tanf(cam.fov[1] * 0.5f) * cam.nearPlane;

    // Camera Space pixel sizes
    Vector2 delta = Vector2((widthHalf * 2.0f) / static_cast<float>(resolution[0] * samplePerLocation),
                            (heightHalf * 2.0f) / static_cast<float>(resolution[1] * samplePerLocation));

    // Camera Vector Correction
    Vector3 gaze = cam.gazePoint - cam.position;
    Vector3 right = Cross(gaze, cam.up).Normalize();
    Vector3 up = Cross(right, gaze).Normalize();
    gaze = Cross(up, right).Normalize();

    // Camera parameters
    Vector3 bottomLeft = cam.position
                        - right *  widthHalf
                        - up * heightHalf
                        + gaze * cam.nearPlane;
    Vector3 pos = cam.position;

    // Kernel Grid-Stride Loop
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalWorkCount;
        threadId += (blockDim.x * gridDim.x))
    {
        Vector2i threadId2d = Vector2i(threadId % (pixelCount[0] * samplePerLocation),
                                       threadId / (pixelCount[0] * samplePerLocation));
        Vector2i globalSampleId = (pixelStart * samplePerLocation) + threadId2d;
        Vector2i globalPixelId = pixelStart + (threadId2d / samplePerLocation);

        // Create random location over sample rectangle
        Vector2 randomOffset = (antiAliasOn)
            ? Vector2(GPUDistribution::Uniform<float>(rng),
                      GPUDistribution::Uniform<float>(rng))
            : Vector2(0.5f);

        // Ray's world position over canvas
        Vector2 sampleDistance = Vector2(static_cast<float>(globalSampleId[0]),
                                         static_cast<float>(globalSampleId[1])) * delta;
        Vector2 samplePointDistance = sampleDistance + randomOffset * delta;
        Vector3 samplePoint = bottomLeft + (samplePointDistance[0] * right)
                                         + (samplePointDistance[1] * up);
        Vector3 rayDir = (samplePoint - pos).Normalize();

        // Generate Required Parameters
        Vector2i pixelSampleId = threadId2d % samplePerLocation;
        Vector2i localPixelId = globalPixelId - pixelStart;
        uint32_t pixelIdLinear = localPixelId[1] * pixelCount[0] + localPixelId[0];
        uint32_t sampleIdLinear = pixelSampleId[1] * samplePerLocation + pixelSampleId[0];

        // Initialize Ray
        RayReg ray;
        ray.ray = RayF(rayDir, pos);
        ray.tMin = cam.nearPlane;
        ray.tMax = cam.farPlane;
        ray.Update(gRays, threadId);

        // Initialize Auxiliary Data
        auxInitF(gAuxiliary[threadId],
                 // Input
                 ray,
                 // Index
                 cam.mediumIndex,
                 pixelIdLinear,
                 sampleIdLinear);

        // Initialize Samples
        ImageAddSample(imgMem, pixelIdLinear, 1);
    }
}

template<class RayAuxData, class AuxInitFunctor>
__device__ __forceinline__
void GenerateCameraRaysGPU(// Output
                           RayGMem* gRays,
                           RayAuxData* gAuxiliary,
                           ImageGMem<Vector4f> imgMem,
                           // Input
                           RNGGMem gRNGStates,
                           const GPUCameraI& gCamera,
                           const uint32_t samplePerLocation,
                           const Vector2i resolution,
                           const Vector2i pixelStart,
                           const Vector2i pixelCount,
                           // Functor to initialize auxiliary base data
                           const AuxInitFunctor auxInitF,
                           // Options
                           bool antiAliasOn)
{
    RandomGPU rng(gRNGStates, LINEAR_GLOBAL_ID);

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
        //ImageAddSample(imgMem, pixelIdLinear, 1);
    }
}

// Templated Camera Ray Generation Kernel
template<class RayAuxData, class AuxInitFunctor>
__global__ CUDA_LAUNCH_BOUNDS_1D
void KCGenCameraRaysFromArrayGPU(// Output
                                 RayGMem* gRays,
                                 RayAuxData* gAuxiliary,
                                 ImageGMem<Vector4f> imgMem,
                                 // Input
                                 RNGGMem gRNGStates,
                                 const GPUCameraI** gCameras,
                                 const uint32_t sceneCamId,
                                 const uint32_t samplePerLocation,
                                 const Vector2i resolution,
                                 const Vector2i pixelStart,
                                 const Vector2i pixelCount,
                                 // Functor to initialize auxiliary base data
                                 const AuxInitFunctor auxInitF,
                                 // Options
                                 bool antiAliasOn)
{
    // Fetch Camera
    const GPUCameraI* gCam = gCameras[sceneCamId];

    GenerateCameraRaysGPU(// Output
                          gRays,
                          gAuxiliary,
                          imgMem,
                          // Input
                          gRNGStates,
                          *gCam,
                          samplePerLocation,
                          resolution,
                          pixelStart,
                          pixelCount,
                          // Functor to initialize auxiliary base data
                          auxInitF,
                          // Options
                          antiAliasOn);
}

// Templated Camera Ray Generation Kernel
template<class RayAuxData, class AuxInitFunctor>
__global__ CUDA_LAUNCH_BOUNDS_1D
void KCGenCameraRaysFromObjectGPU(// Output
                                  RayGMem* gRays,
                                  RayAuxData* gAuxiliary,
                                  ImageGMem<Vector4f> imgMem,
                                  // Input
                                  RNGGMem gRNGStates,
                                  const GPUCameraI& gCamera,
                                  const uint32_t samplePerLocation,
                                  const Vector2i resolution,
                                  const Vector2i pixelStart,
                                  const Vector2i pixelCount,
                                  // Functor to initialize auxiliary base data
                                  const AuxInitFunctor auxInitF,
                                  // Options
                                  bool antiAliasOn)
{
    GenerateCameraRaysGPU(// Output
                          gRays,
                          gAuxiliary,
                          imgMem,
                          // Input
                          gRNGStates,
                          gCamera,
                          samplePerLocation,
                          resolution,
                          pixelStart,
                          pixelCount,
                          // Functor to initialize auxiliary base data
                          auxInitF,
                          // Options
                          antiAliasOn);
}