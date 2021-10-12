#pragma once

#include "MaterialDataStructs.h"
#include "RayLib/Constants.h"
#include "MetaMaterialFunctions.cuh"

#include "GPUSurface.h"

class RandomGPU;

struct BaryMatFuncs
{
    __device__ __forceinline__ static
    Vector3 Sample(// Sampled Output
                   RayF& wo,
                   float& pdf,
                   const GPUMediumI*& outMedium,
                   // Input
                   const Vector3& wi,
                   const Vector3& pos,
                   const GPUMediumI& m,
                   //
                   const BarySurface& surface,
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
    
    __device__ __forceinline__ static
    Vector3 Evaluate(// Input
                     const Vector3& wo,
                     const Vector3& wi,
                     const Vector3& pos,
                     const GPUMediumI& m,
                     //
                     const BarySurface& surface,
                     // Constants
                     const NullData& matData,
                     const HitKey::Type& matId)
    {
        return surface.baryCoords;
    }

    static constexpr auto& Pdf          = PdfOne<NullData, BarySurface>;
    static constexpr auto& Emit         = EmitEmpty<NullData, BarySurface>;
    static constexpr auto& IsEmissive   = IsEmissiveFalse<NullData>;
    static constexpr auto& Specularity  = SpecularityDiffuse<NullData, BarySurface>;
};

struct SphericalMatFuncs
{
    __device__ __forceinline__ static
    Vector3 Sample(// Sampled Output
                   RayF& wo,
                   float& pdf,
                   const GPUMediumI*& outMedium,
                   // Input
                   const Vector3& wi,
                   const Vector3& pos,
                   const GPUMediumI& m,
                   //
                   const SphrSurface& surface,
                   // I-O
                   RandomGPU& rng,
                   // Constants
                   const NullData& matData,
                   const HitKey::Type& matId,
                   uint32_t sampleIndex)
    {
        // No medium change
        outMedium = &m;
        pdf = 1.0f;
        wo = InvalidRayF;
        return Vector3f(surface.sphrCoords[0],
                        surface.sphrCoords[1],
                        0.0f);
    }
    
    __device__ __forceinline__ static
    Vector3 Evaluate(// Input
                     const Vector3& wo,
                     const Vector3& wi,
                     const Vector3& pos,
                     const GPUMediumI& m,
                     //
                     const SphrSurface& surface,
                     // Constants
                     const NullData& matData,
                     const HitKey::Type& matId)
    {
        return Vector3f(surface.sphrCoords[0],
                        surface.sphrCoords[1],
                        0.0f);
    }

    static constexpr auto& Pdf          = PdfOne<NullData, SphrSurface>;
    static constexpr auto& Emit         = EmitEmpty<NullData, SphrSurface>;
    static constexpr auto& IsEmissive   = IsEmissiveFalse<NullData>;
    static constexpr auto& Specularity  = SpecularityDiffuse<NullData, SphrSurface>;

};

struct NormalRenderMatFuncs
{
    __device__ __forceinline__ static
    Vector3 Sample(// Sampled Output
                   RayF& wo,
                   float& pdf,
                   const GPUMediumI*& outMedium,
                   // Input
                   const Vector3& wi,
                   const Vector3& pos,
                   const GPUMediumI& m,
                   //
                   const BasicSurface& surface,
                   // I-O
                   RandomGPU& rng,
                   // Constants
                   const NullData& matData,
                   const HitKey::Type& matId,
                   uint32_t sampleIndex)
    {
        // No medium change
        outMedium = &m;
        pdf = 1.0f;
        wo = InvalidRayF;
    
        Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
        normal = 0.5f * normal + 0.5f;
    
        return normal;
    }
    
    __device__ __forceinline__ static
    Vector3 Evaluate(// Input
                     const Vector3& wo,
                     const Vector3& wi,
                     const Vector3& pos,
                     const GPUMediumI& m,
                     //
                     const BasicSurface& surface,
                     // Constants
                     const NullData& matData,
                     const HitKey::Type& matId)
    {
        Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
        normal = 0.5f * normal + 0.5f;
    
        return normal;
    }

    static constexpr auto& Pdf          = PdfOne<NullData, BasicSurface>;
    static constexpr auto& Emit         = EmitEmpty<NullData, BasicSurface>;
    static constexpr auto& IsEmissive   = IsEmissiveFalse<NullData>;
    static constexpr auto& Specularity  = SpecularityDiffuse<NullData, BasicSurface>;
};