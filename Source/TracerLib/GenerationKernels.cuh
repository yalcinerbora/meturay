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

// Templated Camera Ray Generation Kernel
template<class RayAuxData, AuxInitFunc<RayAuxData> AuxFunc>
__global__ void KCGenerateCameraRaysCPU(// Output
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
                                        // Data to initialize auxiliary base data
                                        const RayAuxData auxBaseData)
{
    RandomGPU rng(gRNGStates.state, LINEAR_GLOBAL_ID);

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
        float dX = GPUDistribution::Uniform<float>(rng);
        float dY = GPUDistribution::Uniform<float>(rng);
        Vector2 randomOffset = Vector2(dX, dY);
        //Vector2 randomOffset = Vector2(0.5f);

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
        AuxFunc(gAuxiliary[threadId],
                // Input
                auxBaseData,
                ray,
                // Index
                cam.mediumIndex,
                pixelIdLinear,
                sampleIdLinear);


        // Initialize Samples
        ImageAddSample(imgMem, pixelIdLinear, 1);

    }
}

// Templated Camera Ray Generation Kernel
template<class RayAuxData, AuxInitFunc<RayAuxData> AuxFunc>
__global__ void KCGenerateCameraRaysGPU(// Output
                                        RayGMem* gRays,
                                        RayAuxData* gAuxiliary,
                                        ImageGMem<Vector4f> imgMem,
                                        // Input
                                        RNGGMem gRNGStates,
                                        const GPUCameraI* gCam,
                                        const uint32_t samplePerLocation,
                                        const Vector2i resolution,
                                        const Vector2i pixelStart,
                                        const Vector2i pixelCount,
                                        // Data to initialize auxiliary base data
                                        const RayAuxData auxBaseData)
{
    RandomGPU rng(gRNGStates.state, LINEAR_GLOBAL_ID);

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
        gCam->GenerateRay(ray,
                         //
                         globalSampleId,
                         totalSamples,
                         rng);
        // Generate Required Parameters
        Vector2i pixelSampleId = threadId2d % samplePerLocation;
        Vector2i localPixelId = globalPixelId - pixelStart;
        uint32_t pixelIdLinear = localPixelId[1] * pixelCount[0] + localPixelId[0];
        uint32_t sampleIdLinear = pixelSampleId[1] * samplePerLocation + pixelSampleId[0];

        // Write Ray
        ray.Update(gRays, threadId);

        // Write Auxiliary Data
        AuxFunc(gAuxiliary[threadId],
                // Input
                auxBaseData,
                ray,
                // Index
                gCam->MediumIndex(),
                pixelIdLinear,
                sampleIdLinear);

        // Initialize Samples
        ImageAddSample(imgMem, pixelIdLinear, 1);
    }
}