#pragma once

/**

Camera Ray Generation Kernel

Uses statified sampling

*/

#include <cstdint>
#include <cuda_runtime.h>

#include "RayLib/Vector.h"
#include "RayLib/Camera.h"

#include "RayStructs.h"
#include "Random.cuh"

// Commands that initialize ray auxiliary data
template<class RayAuxData>
using AuxInitFunc = void(*)(RayAuxData*,
                            const uint32_t writeLoc,
                            // Input
                            const RayAuxData&,
                            const RayReg&,
                            // Index
                            const uint32_t localPixelId,
                            const uint32_t pixelSampleId);

// Templated Camera Ray Generation Kernel
template<class RayAuxData, AuxInitFunc<RayAuxData> AuxFunc>
__global__ void KCGenerateCameraRays(// Output
                                     RayGMem* gRays,
                                     RayAuxData* gAuxiliary,
                                     // Input
                                     RNGGMem gRNGStates,
                                     const CameraPerspective cam,
                                     const uint32_t samplePerLocation,
                                     const Vector2i resolution,
                                     const Vector2i pixelStart,
                                     const Vector2i pixelCount,
                                     // Data to initialize auxiliary base data
                                     const RayAuxData auxBaseData)
{
    extern __shared__ uint32_t sRandState[];
    RandomGPU rng(gRNGStates.state, sRandState);

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
        float dX = GPURand::ZeroOne<float>(rng);
        float dY = GPURand::ZeroOne<float>(rng);
        //Vector2 randomOffset = Vector2(dX, dY);
        Vector2 randomOffset = Vector2(0.5f);

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
        ray.tMin = 0; // cam.nearPlane;
        ray.tMax = FLT_MAX; // cam.farPlane;
        ray.Update(gRays, threadId);

        // Initialize Auxiliary Data
        AuxFunc(gAuxiliary,
                threadId,
                // Input
                auxBaseData,
                ray,
                // Index
                pixelIdLinear,
                sampleIdLinear);
    }
}