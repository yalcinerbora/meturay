#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/HemiDistribution.h"

#include "RNGenerator.h"
#include "MaterialFunctions.h"
#include "GPUSurface.h"
#include "MetaMaterialFunctions.cuh"

struct LambertMatFuncs
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
                   const UVSurface& surface,
                   // I-O
                   RNGeneratorGPUI& rng,
                   // Constants
                   const LambertMatData& matData,
                   const HitKey::Type& matId,
                   uint32_t sampleIndex)
    {
        // No medium change
        outMedium = &m;

        // Ray Selection
        const Vector3& position = pos;
        Vector3 normal = ZAxis;
        // Check if tangent space normal is avail
        if(matData.dNormal[matId])
            normal = (*matData.dNormal[matId])(surface.uv).Normalize();

        // Generate New Ray Direction (This is in tangent space)
        Vector2 xi(rng.Uniform(), rng.Uniform());
        Vector3 direction = HemiDistribution::HemiCosineCDF(xi, pdf);
        direction.NormalizeSelf();

        // Cos Theta
        float nDotL = max(normal.Dot(direction), 0.0f);
        // Ray out
        wo = RayF(GPUSurface::ToSpace(direction, surface.worldToTangent), position);
        wo.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);

        // Radiance Calculation
        const Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
        //printf("%f, %f, %f\n", albedo[0], albedo[1], albedo[2]);
        return nDotL * albedo * MathConstants::InvPi;
    }

    __device__ inline static
    float Pdf(const Vector3& wo,
              const Vector3& wi,
              const Vector3& pos,
              const GPUMediumI& m,
              //
              const UVSurface& surface,
              // Constants
              const LambertMatData& matData,
              const HitKey::Type& matId)
    {
        Vector3 normal = ZAxis;
        // Check if tangent space normal is avail
        if(matData.dNormal[matId])
            normal = (*matData.dNormal[matId])(surface.uv).Normalize();
        normal = GPUSurface::ToSpace(normal, surface.worldToTangent);

        float pdf = max(wo.Dot(normal), 0.0f);
        pdf *= MathConstants::InvPi;
        return pdf;
    }

    __device__ inline static
    Vector3 Evaluate(// Input
                     const Vector3& wo,
                     const Vector3& wi,
                     const Vector3& pos,
                     const GPUMediumI& m,
                     //
                     const UVSurface& surface,
                     // Constants
                     const LambertMatData& matData,
                     const HitKey::Type& matId)
    {
        Vector3 normal = ZAxis;
        // Check if tangent space normal is avail
        if(matData.dNormal[matId])
            normal = (*matData.dNormal[matId])(surface.uv).Normalize();

        // Calculate lightning in world space since
        // wo is already in world space
        normal = GPUSurface::ToSpace(normal, surface.worldToTangent);

        float nDotL = max(normal.Dot(wo), 0.0f);
        const Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
        return nDotL * albedo * MathConstants::InvPi;
    }

    // Does not have emission
    static constexpr auto& IsEmissive   = IsEmissiveFalse<LambertMatData>;
    static constexpr auto& Emit         = EmitEmpty<LambertMatData, UVSurface>;
    static constexpr auto& Specularity  = SpecularityDiffuse<LambertMatData, UVSurface>;
};