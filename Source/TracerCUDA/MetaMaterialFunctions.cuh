#pragma once

#include <cuda.h>

#include "RayLib/HitStructs.h"
#include "RayLib/Ray.h"

class GPUMediumI;
struct TexCoords;
class RNGeneratorGPUI;

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
                  const Vector3&,
                  const Vector3&,
                  const GPUMediumI&,
                  //
                  const Surface&,
                  // Constants
                  const Data&,
                  const HitKey::Type&)
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
                    const Vector3&,
                    const Vector3&,
                    const GPUMediumI& m,
                    //
                    const Surface&,
                    // I-O
                    RNGeneratorGPUI&,
                    // Constants
                    const Data&,
                    const HitKey::Type&,
                    uint32_t)
{
    outMedium = &m;
    wo = InvalidRayF;
    pdf = 0.0f;
    return Zero3f;
}

template<class Data, class Surface>
__device__ inline
float PdfZero(const Vector3&,
              const Vector3&,
              const Vector3&,
              const GPUMediumI&,
              //
              const Surface&,
              // Constants
              const Data&,
              const HitKey::Type&)
{
    return 0.0f;
}

template<class Data, class Surface>
__device__ inline
float PdfOne(const Vector3&,
             const Vector3&,
             const Vector3&,
             const GPUMediumI&,
             //
             const Surface&,
             // Constants
             const Data&,
             const HitKey::Type&)
{
    return 1.0f;
}

template <class Data, class Surface>
__device__ inline
Vector3 EvaluateEmpty(// Input
                      const Vector3&,
                      const Vector3&,
                      const Vector3&,
                      const GPUMediumI&,
                      //
                      const Surface&,
                      // Constants
                      const Data&,
                      const HitKey::Type&)
{
    return Zero3;
}

template <class Data, class Surface>
__device__ inline
float SpecularityPerfect(const Surface&, const Data&,
                         const HitKey::Type&)
{
    return 1.0f;
}

template <class Data, class Surface>
__device__ inline
float SpecularityDiffuse(const Surface&,
                         const Data&,
                         const HitKey::Type&)
{
    return 0.0f;
}