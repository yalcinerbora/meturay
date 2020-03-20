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
#include "ImageStructs.h"
#include "Random.cuh"
#include "GPUEndpointI.cuh"

#include "ImageFunctions.cuh"

// Commands that initialize ray auxiliary data
template<class RayAuxData>
using AuxInitFunc = void(*)(RayAuxData&,
                            // Input
                            const RayAuxData&,
                            const RayReg&,
                            // Index
                            const uint32_t localPixelId,
                            const uint32_t pixelSampleId);

// Templated Camera Ray Generation Kernel
template<class RayAuxData, AuxInitFunc<RayAuxData> AuxFunc>
__global__ void KCGenerateCameraRaysCPU(// Output
                                        RayGMem* gRays,
                                        RayAuxData* gAuxiliary,
                                        ImageGMem<Vector4f> imgMem,
                                        // Input
                                        RNGGMem gRNGStates,
                                        const CPUCamera cam,
                                        const uint32_t samplePerLocation,
                                        const Vector2i resolution,
                                        const Vector2i pixelStart,
                                        const Vector2i pixelCount,
                                        // Data to initialize auxiliary base data
                                        const RayAuxData auxBaseData)
{
    RandomGPU rng(gRNGStates.state);

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
        AuxFunc(gAuxiliary,
                threadId,
                // Input
                auxBaseData,
                ray,
                // Index
                pixelIdLinear,
                sampleIdLinear);

        // Initialize Samples
        ImageAddSample(imgMem, pixelIdLinear, 1);

    }
}

#include "GPUCamera.cuh"

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
    RandomGPU rng(gRNGStates.state);

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
        /*if(threadId == 0) printf("%p\n", *gCam);*/
        //printf("p %f, %f, %f --- d %f, %f, %f\n",
        //       ray.ray.getPosition()[0], ray.ray.getPosition()[1], ray.ray.getPosition()[2],
        //       ray.ray.getDirection()[0], ray.ray.getDirection()[1], ray.ray.getDirection()[2]);
        //if(threadId == 0)
        //{
        //    printf("0x%llx\n", *reinterpret_cast<const uint64_t*>(gCam));
        //    printf("0x%llx\n", *(reinterpret_cast<const uint64_t*>(gCam) + 1));
        //    printf("0X%llx\n", *(reinterpret_cast<const uint64_t*>(gCam) + 2));
        //}
        //auto cammo = static_cast<const PinholeCamera*>(gCam);
        //cammo->GenerateRay(ray,
        //                   //
        //                   globalSampleId,
        //                   totalSamples,
        //                   rng);
        //if(threadId == 0)
        //    printf("pos %f %f %f\n", 
        //           cammo->position[0],
        //           cammo->position[1],
        //           cammo->position[2]);

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
                pixelIdLinear,
                sampleIdLinear);

        // Initialize Samples
        ImageAddSample(imgMem, pixelIdLinear, 1);
    }
}