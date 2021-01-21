#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"

#include "Random.cuh"
#include "TextureStructs.h"
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
                      const TexCoords* uvs,
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
                       const TexCoords* uvs,
                       // Constants
                       const UnrealMatData& matData,
                       const HitKey::Type& matId)
{
    Vector3 N = GPUSurface::NormalWorld(surface.worldToTangent);

    return Zero3;
}