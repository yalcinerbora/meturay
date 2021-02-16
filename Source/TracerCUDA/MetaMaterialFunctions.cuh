#pragma once

#include <cuda.h>

#include "RayLib/HitStructs.h"
#include "RayLib/Ray.h"

class GPUMediumI;
class RandomGPU;
struct TexCoords;

template <class Data>
__device__ inline
bool IsEmissiveFalse(const Data&,
                     const HitKey::Type&)
{
    return false;
}

template <class Data>
__device__ inline
bool IsEmissiveTrue(const Data&,
                    const HitKey::Type&)
{
    return true;
}

template <class Data, class Surface>
__device__ inline
Vector3 EmitEmpty(// Input
                  const Vector3& wo,
                  const Vector3& pos,
                  const GPUMediumI& m,
                  //
                  const Surface& surface,
                  // Constants
                  const Data&,
                  const HitKey::Type& matId)
{
    return Zero3;
}

template <class Data, class Surface>
__device__ inline
Vector3 SampleEmpty(// Sampled Output
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
                    const Data& matData,
                    const HitKey::Type& matId,
                    uint32_t sampleIndex)
{
    return Zero3f;
}

template <class Data, class Surface>
__device__ inline
Vector3 EvaluateEmpty(// Input
                         const Vector3& wo,
                         const Vector3& wi,
                         const Vector3& pos,
                         const GPUMediumI& m,
                         //
                         const Surface& surface,
                         // Constants
                         const Data& matData,
                         const HitKey::Type& matId)
{
    return Zero3;
}