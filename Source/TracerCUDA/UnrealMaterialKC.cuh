#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"

#include "Random.cuh"
#include "ImageFunctions.cuh"
#include "MaterialFunctions.cuh"
#include "TracerFunctions.cuh"
#include "GPUSurface.h"

__device__ inline
Vector3 UnrealSample(// Sampled Output
                      RayF& wo,
                      float& pdf,
                      const GPUMediumI*& outMedium,
                      // Input
                      const Vector3& wi,
                      const Vector3& pos,
                      const GPUMediumI& m,
                      //
                      const UVSurface& surface,
                      // I-O
                      RandomGPU& rng,
                      // Constants
                      const UnrealMatData& matData,
                      const HitKey::Type& matId,
                      uint32_t sampleIndex)
{
    // No medium change
    outMedium = &m;

    Vector3 N = GPUSurface::NormalWorld(surface.worldToTangent);

    return Zero3;
}

__device__ inline
Vector3 UnrealEvaluate(// Input
                       const Vector3& wo,
                       const Vector3& wi,
                       const Vector3& pos,
                       const GPUMediumI& m,
                       //
                       const UVSurface& surface,
                       // Constants
                       const UnrealMatData& matData,
                       const HitKey::Type& matId)
{
    Vector3 N = GPUSurface::NormalWorld(surface.worldToTangent);

    return Zero3;
}