#pragma once

#include <cuda.h>

#include "RayLib/HitStructs.h"
#include "RayLib/Ray.h"

class GPUMediumI;
class RandomGPU;
struct TexCoords;

template <class Data>
__device__ __forceinline__
bool IsEmissiveFalse(const Data&,
                     const HitKey::Type&)
{
    return false;
}

template <class Data>
__device__ __forceinline__
bool IsEmissiveTrue(const Data&,
                    const HitKey::Type&)
{
    return true;
}

template <class Data, class Surface>
__device__ __forceinline__
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
__device__ __forceinline__
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
    outMedium = &m;
    wo = InvalidRayF;
    pdf = 0.0f;
    return Zero3f;
}

template<class Data, class Surface>
__device__ __forceinline__
float PdfZero(const Vector3&,
              const Vector3&,
              const Vector3&,
              const GPUMediumI&,
              //
              const Surface&,
              // Constants
              const Data& matData,
              const HitKey::Type& matId)
{
    return 0.0f;
}

template<class Data, class Surface>
__device__ __forceinline__
float PdfOne(const Vector3&,
             const Vector3&,
             const Vector3&,
             const GPUMediumI&,
             //
             const Surface&,
             // Constants
             const Data& matData,
             const HitKey::Type& matId)
{
    return 1.0f;
}

template <class Data, class Surface>
__device__ __forceinline__
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

template <class Data, class Surface>
__device__ __forceinline__
float SpecularityPerfect(const Surface&, 
                         const Data& matData,
                         const HitKey::Type& matId)
{
    return 1.0f;
}

template <class Data, class Surface>
__device__ __forceinline__
float SpecularityDiffuse(const Surface&,
                         const Data& matData,
                         const HitKey::Type& matId)
{
    return 0.0f;
}