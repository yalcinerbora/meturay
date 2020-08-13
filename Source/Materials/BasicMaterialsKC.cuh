#pragma once

struct RayReg;
class RandomGPU;

#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"

#include "TracerLib/ImageFunctions.cuh"
#include "TracerLib/SurfaceStructs.h"
#include "TracerLib/TextureStructs.h"

__device__ inline
Vector3 ConstantSample(// Sampled Output
                       RayF& wo,
                       float& pdf,
                       GPUMedium& outMedium,
                       // Input
                       const Vector3& wi,
                       const Vector3& pos,
                       const GPUMedium& m,
                       //
                       const EmptySurface& surface,
                       const TexCoords* uvs,
                       // I-O
                       RandomGPU& rng,
                       // Constants
                       const AlbedoMatData& matData,
                       const HitKey::Type& matId,
                       uint32_t sampleIndex)
{
    outMedium = m;
    static constexpr Vector3 ZERO = Zero3;
    pdf = 1.0f;
    wo = RayF(ZERO, ZERO);
    return matData.dAlbedo[matId];
}

__device__ inline
Vector3 ConstantEvaluate(// Input
                         const Vector3& wo,
                         const Vector3& wi,
                         const Vector3& pos,
                         const GPUMedium& m,
                         //
                         const EmptySurface& surface,
                         const TexCoords* uvs,
                         // Constants
                         const AlbedoMatData& matData,
                         const HitKey::Type& matId)
{
    return matData.dAlbedo[matId];
}

__device__ inline
Vector3 BarycentricSample(// Sampled Output
                          RayF& wo,
                          float& pdf,
                          GPUMedium& outMedium,
                          // Input
                          const Vector3& wi,
                          const Vector3& pos,
                          const GPUMedium& m,
                          //
                          const BarySurface& surface,
                          const TexCoords* uvs,
                          // I-O
                          RandomGPU& rng,
                          // Constants
                          const NullData& matData,
                          const HitKey::Type& matId,
                          uint32_t sampleIndex)
{
    outMedium = m;
    static constexpr Vector3 ZERO = Zero3;
    pdf = 1.0f;
    wo = RayF(ZERO, ZERO);
    return surface.baryCoords;
}

__device__ inline
Vector3 BarycentricEvaluate(// Input
                            const Vector3& wo,
                            const Vector3& wi,
                            const Vector3& pos,
                            const GPUMedium& m,
                            //
                            const BarySurface& surface,
                            const TexCoords* uvs,
                            // Constants
                            const NullData& matData,
                            const HitKey::Type& matId)
{
    return surface.baryCoords;
}

__device__ inline
Vector3 SphericalSample(// Sampled Output
                        RayF& wo,
                        float& pdf,
                        GPUMedium& outMedium,
                        // Input
                        const Vector3& wi,
                        const Vector3& pos,
                        const GPUMedium& m,
                        //
                        const SphrSurface& surface,
                        const TexCoords* uvs,
                        // I-O
                        RandomGPU& rng,
                        // Constants
                        const NullData& matData,
                        const HitKey::Type& matId,
                        uint32_t sampleIndex)
{
    outMedium = m;
    static constexpr Vector3 ZERO = Zero3;
    pdf = 1.0f;
    wo = RayF(ZERO, ZERO);
    return Vector3f(cos(surface.sphrCoords[0]),
                    sin(surface.sphrCoords[1]),
                    0.0f);
}

__device__ inline
Vector3 SphericalEvaluate(// Input
                          const Vector3& wo,
                          const Vector3& wi,
                          const Vector3& pos,
                          const GPUMedium& m,
                          //
                          const SphrSurface& surface,
                          const TexCoords* uvs,
                          // Constants
                          const NullData& matData,
                          const HitKey::Type& matId)
{
    return Vector3f(cos(surface.sphrCoords[0]),
                    sin(surface.sphrCoords[1]),
                    0.0f);
}