#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/HemiDistribution.h"

#include "Random.cuh"
#include "MaterialFunctions.cuh"
#include "GPUSurface.h"

__device__ inline
Vector3 LambertTSample(// Sampled Output
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
                       RandomGPU& rng,
                       // Constants
                       const LambertTMatData& matData,
                       const HitKey::Type& matId,
                       uint32_t sampleIndex)
{
    // No medium change
    outMedium = &m;

    // Ray Selection
    const Vector3& position = pos;

    // Check if tangent space normal is avail
    Vector3 normal;
    if(matData.dNormal[matId])
    {
        const auto& dNormalSampler = *matData.dNormal[matId];
        normal = GPUSurface::ToWorld(dNormalSampler(surface.uv),
                                     surface.worldToTangent);
    }    
    // Tangent space normal not present (mat has not tangent space normal)
    // use interpolated normal
    else normal = GPUSurface::NormalWorld(surface.worldToTangent);
    
    // Generate New Ray Direction
    Vector2 xi(GPUDistribution::Uniform<float>(rng),
               GPUDistribution::Uniform<float>(rng));
    Vector3 direction = HemiDistribution::HemiCosineCDF(xi, pdf);
    //Vector3 direction = HemiDistribution::HemiUniformCDF(xi, pdf);
    direction.NormalizeSelf();

    // Generated direction vector is on surface space (hemisperical)
    // Convert it to normal oriented hemisphere (world space)
    QuatF q = Quat::RotationBetweenZAxis(normal);
    direction = q.ApplyRotation(direction);

    // Cos Tetha
    float nDotL = max(normal.Dot(direction), 0.0f);
    
    // Ray out
    Vector3 outPos = position + normal * MathConstants::Epsilon;
    wo = RayF(direction, outPos);
    // BSDF Calculation
    const Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
    return nDotL * albedo * MathConstants::InvPi;
}

__device__ inline
Vector3 LambertTEvaluate(// Input
                         const Vector3& wo,
                         const Vector3& wi,
                         const Vector3& pos,
                         const GPUMediumI& m,
                         //
                         const UVSurface& surface,
                         // Constants
                         const LambertTMatData& matData,
                         const HitKey::Type& matId)
{
    // Check if tangent space normal is avail
    Vector3 normal;
    if(matData.dNormal[matId])
    {
        const auto& dNormalSampler = *matData.dNormal[matId];
        normal = GPUSurface::ToWorld(dNormalSampler(surface.uv),
                                     surface.worldToTangent);
    }
    // Tangent space normal not present (mat has not tangent space normal)
    // use interpolated normal
    else normal = GPUSurface::NormalWorld(surface.worldToTangent);

    float nDotL = max(normal.Dot(wo), 0.0f);
       
    const Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
    return nDotL * albedo * MathConstants::InvPi;
}