#pragma once

struct RayReg;
class RandomGPU;

#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"

#include "ImageFunctions.cuh"
#include "GPUSurface.h"
#include "TextureStructs.h"

__device__ inline
Vector3 ConstantSample(// Sampled Output
                       RayF& wo,
                       float& pdf,
                       const GPUMediumI*& outMedium,
                       // Input
                       const Vector3& wi,
                       const Vector3& pos,
                       const GPUMediumI& m,
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
    // No medium change
    outMedium = &m;

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
                         const GPUMediumI& m,
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
                          const GPUMediumI*& outMedium,
                          // Input
                          const Vector3& wi,
                          const Vector3& pos,
                          const GPUMediumI& m,
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
    // No medium change
    outMedium = &m;

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
                            const GPUMediumI& m,
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
                        const GPUMediumI*& outMedium,
                        // Input
                        const Vector3& wi,
                        const Vector3& pos,
                        const GPUMediumI& m,
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
    // No medium change
    outMedium = &m;

    static constexpr Vector3 ZERO = Zero3;
    pdf = 1.0f;
    wo = RayF(ZERO, ZERO);
    return Vector3f(surface.sphrCoords[0],
                    surface.sphrCoords[1],
                    0.0f);
}

__device__ inline
Vector3 SphericalEvaluate(// Input
                          const Vector3& wo,
                          const Vector3& wi,
                          const Vector3& pos,
                          const GPUMediumI& m,
                          //
                          const SphrSurface& surface,
                          const TexCoords* uvs,
                          // Constants
                          const NullData& matData,
                          const HitKey::Type& matId)
{
    return Vector3f(surface.sphrCoords[0],
                    surface.sphrCoords[1],
                    0.0f);
}

__device__ inline
Vector3 NormalSample(// Sampled Output
                     RayF& wo,
                     float& pdf,
                     const GPUMediumI*& outMedium,
                     // Input
                     const Vector3& wi,
                     const Vector3& pos,
                     const GPUMediumI& m,
                     //
                     const BasicSurface& surface,
                     const TexCoords* uvs,
                     // I-O
                     RandomGPU& rng,
                     // Constants
                     const NullData& matData,
                     const HitKey::Type& matId,
                     uint32_t sampleIndex)
{
    // No medium change
    outMedium = &m;
    static constexpr Vector3 ZERO = Zero3;
    pdf = 1.0f;
    wo = RayF(ZERO, ZERO);

    Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
    normal = 0.5f * normal + 0.5f;

    return normal;
}

__device__ inline
Vector3 NormalEvaluate(// Input
                       const Vector3& wo,
                       const Vector3& wi,
                       const Vector3& pos,
                       const GPUMediumI& m,
                       //
                       const BasicSurface& surface,
                       const TexCoords* uvs,
                       // Constants
                       const NullData& matData,
                       const HitKey::Type& matId)
{
    Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
    normal = 0.5f * normal + 0.5f;

    return normal;
}