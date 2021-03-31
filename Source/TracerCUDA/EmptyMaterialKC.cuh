#pragma once

#include "MaterialDataStructs.h"

#include "GPUSurface.h"

class RandomGPU;

template <class Surface>
__device__
Vector3 EmptySample(// Sampled Output
                       RayF& wo,
                       float& pdf,
                       const GPUMediumI*& outMedium,
                       // Input
                       const Vector3& wi,
                       const Vector3& pos,
                       const GPUMediumI& m,
                       //
                       const Surface& surface,
                       // I-O
                       RandomGPU& rng,
                       // Constants
                       const NullData& matData,
                       const HitKey::Type& matId,
                       uint32_t sampleIndex)
{
    static constexpr Vector3 ZERO = Zero3;
    wo = RayF(ZERO, ZERO);
    outMedium = &m;
    pdf = 1.0f;
    return ZERO;
}

template <class Surface>
__device__
Vector3 EmptyEvaluate(// Input
                      const Vector3& wo,
                      const Vector3& wi,
                      const Vector3& pos,
                      const GPUMediumI& m,
                      //
                      const Surface& surface,
                      // Constants
                      const NullData& matData,
                      const HitKey::Type& matId)
{
    return Zero3;
}