#include "GPUCamera.cuh"
#include "Random.cuh"
#include "RayStructs.h"

__device__ 
PinholeCamera::PinholeCamera(const CPUCamera& cam, HitKey k)
    : GPUEndpointI(k)
{
    // Find world space window sizes
    float widthHalf = tanf(cam.fov[0] * 0.5f) * cam.nearPlane;
    float heightHalf = tanf(cam.fov[1] * 0.5f) * cam.nearPlane;

    // Camera Vector Correction
    Vector3 gaze = cam.gazePoint - cam.position;
    right = Cross(gaze, cam.up).Normalize();
    up = Cross(right, gaze).Normalize();
    gaze = Cross(up, right).Normalize();

    // Camera parameters
    bottomLeft = (cam.position
                  - right * widthHalf
                  - up * heightHalf
                  + gaze * cam.nearPlane);
    position = cam.position;
}

__device__ 
void PinholeCamera::Sample(// Output
                           HitKey& materialKey,
                           Vector3& direction,
                           float& pdf,
                           // Input
                           const Vector3& sampleLoc,
                           // I-O
                           RandomGPU&) const
{
    // One
    materialKey = boundaryMaterialKey;
    direction = sampleLoc - position;
    pdf = 1.0f;
}

__device__ 
void PinholeCamera::GenerateRay(// Output
                                RayReg& ray,
                                // Input
                                const Vector2i& sampleId,
                                const Vector2i& sampleMax,
                                // I-O
                                RandomGPU& rng) const
{
    // DX DY from stratfied sample
    Vector2 delta = Vector2(planeSize[0] / static_cast<float>(sampleMax[0]),
                            planeSize[1] / static_cast<float>(sampleMax[1]));

    // Create random location over sample rectangle
    float dX = GPUDistribution::Uniform<float>(rng);
    float dY = GPUDistribution::Uniform<float>(rng);
    Vector2 randomOffset = Vector2(dX, dY);
    //Vector2 randomOffset = Vector2(0.5f);

    Vector2 sampleDistance = Vector2(static_cast<float>(sampleId[0]),
                                     static_cast<float>(sampleId[1])) * delta;
    sampleDistance += (randomOffset * delta);
    Vector3 samplePoint = bottomLeft + (sampleDistance[0] * right)
                                     + (sampleDistance[1] * up);
    Vector3 rayDir = (samplePoint - position).Normalize();

    // Initialize Ray
    ray.ray = RayF(rayDir, position);
    ray.tMin = nearFar[0];
    ray.tMax = nearFar[1];
}