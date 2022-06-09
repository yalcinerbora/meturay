#pragma once

#include "MaterialDataStructs.h"
#include "RayLib/Constants.h"
#include "MetaMaterialFunctions.cuh"

#include "GPUSurface.h"

#include "RayLib/CoordinateConversion.h"

class RNGeneratorGPUI;

struct BaryMatFuncs
{
    __device__ inline static
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
                   RNGeneratorGPUI& rng,
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

    __device__ inline static
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
    __device__ inline static
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
                   RNGeneratorGPUI& rng,
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

    __device__ inline static
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

__device__ static constexpr Vector3f COLORS[8] =
{
    Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, 1), Vector3f(0, 1, 1),
    Vector3f(1, 0, 1), Vector3f(1, 1, 0), Vector3f(0, 0, 0), Vector3f(0.24, 0.47, 0.43)
};

struct SphericalAnisoTestFuncs
{
    __device__ inline static
    Vector4uc TEST_INTERP(Vector2f& interp,
                          const Vector3f& direction)
    {
        // I couldn't comprehend this as a mathematical
        // representation so tabulated the output
        static constexpr Vector4uc TABULATED_LAYOUTS[12] =
        {
            Vector4uc(0,1,0,1), Vector4uc(1,2,1,2),  Vector4uc(2,3,2,3), Vector4uc(3,0,3,0),
            Vector4uc(0,1,4,5), Vector4uc(1,2,5,6),  Vector4uc(2,3,6,7), Vector4uc(3,0,7,4),
            Vector4uc(4,5,4,5), Vector4uc(5,6,5,6),  Vector4uc(6,7,6,7), Vector4uc(7,4,7,4)
        };

        static constexpr float PIXEL_X = 4;
        static constexpr float PIXEL_Y = 2;

        Vector3 dirZUp = Vector3(direction[2], direction[0], direction[1]);
        Vector2f thetaPhi = Utility::CartesianToSphericalUnit(dirZUp);
        // Normalize to generate UV [0, 1]
        // theta range [-pi, pi]
        float u = (thetaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
        // phi range [0, pi]
        float v = 1.0f - (thetaPhi[1] / MathConstants::Pi);

        // Convert to pixelCoords
        float pixelX = u * PIXEL_X;
        float pixelY = v * PIXEL_Y;

        float indexX;
        float interpX = modff(pixelX, &indexX);
        uint32_t indexXInt = (indexX >= 4) ? 0 : static_cast<uint32_t>(indexX);

        float indexY;
        float interpY = abs(modff(pixelY + 0.5f, &indexY));
        uint32_t indexYInt = static_cast<uint32_t>(indexY);

        interp = Vector2f(interpX, interpY);
        return TABULATED_LAYOUTS[indexYInt * 4 + indexXInt];;
    }

    __device__ inline static
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
                   RNGeneratorGPUI& rng,
                   // Constants
                   const NullData& matData,
                   const  HitKey::Type& matId,
                   uint32_t sampleIndex)
    {
        // No medium change
        outMedium = &m;
        pdf = 1.0f;
        wo = InvalidRayF;

        Vector2f interp;
        Vector4uc indices = TEST_INTERP(interp, surface.WorldGeoNormal());
        Vector3f a = Vector3f::Lerp(COLORS[indices[0]], COLORS[indices[1]], interp[0]);
        Vector3f b = Vector3f::Lerp(COLORS[indices[2]], COLORS[indices[3]], interp[0]);
        Vector3f result = Vector3f::Lerp(a, b, interp[1]);
        return result;
    }

    __device__ inline static
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
        Vector2f interp;
        Vector4uc indices = TEST_INTERP(interp, surface.WorldGeoNormal());
        Vector3f a = Vector3f::Lerp(COLORS[indices[0]], COLORS[indices[1]], interp[0]);
        Vector3f b = Vector3f::Lerp(COLORS[indices[2]], COLORS[indices[3]], interp[0]);
        Vector3f result = Vector3f::Lerp(a, b, interp[1]);
        return result;
    }

    static constexpr auto& Pdf          = PdfOne<NullData, BasicSurface>;
    static constexpr auto& Emit         = EmitEmpty<NullData, BasicSurface>;
    static constexpr auto& IsEmissive   = IsEmissiveFalse<NullData>;
    static constexpr auto& Specularity  = SpecularityDiffuse<NullData, BasicSurface>;

};

struct NormalRenderMatFuncs
{
    __device__ inline static
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
                   RNGeneratorGPUI& rng,
                   // Constants
                   const NullData& matData,
                   const HitKey::Type& matId,
                   uint32_t sampleIndex)
    {
        // No medium change
        outMedium = &m;
        pdf = 1.0f;
        wo = InvalidRayF;

        Vector3 normal = GPUSurface::NormalToSpace(surface.worldToTangent);
        normal = 0.5f * normal + 0.5f;

        return normal;
    }

    __device__ inline static
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
        Vector3 normal = GPUSurface::NormalToSpace(surface.worldToTangent);
        normal = 0.5f * normal + 0.5f;
        return normal;
    }

    static constexpr auto& Pdf          = PdfOne<NullData, BasicSurface>;
    static constexpr auto& Emit         = EmitEmpty<NullData, BasicSurface>;
    static constexpr auto& IsEmissive   = IsEmissiveFalse<NullData>;
    static constexpr auto& Specularity  = SpecularityDiffuse<NullData, BasicSurface>;
};