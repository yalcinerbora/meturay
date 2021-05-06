#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/HemiDistribution.h"

#include "Random.cuh"
#include "MaterialFunctions.cuh"
#include "GPUSurface.h"

__device__ __forceinline__
Vector3 LambertSample(// Sampled Output
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
        normal = (*matData.dNormal[matId])(surface.uv);

    // Generate New Ray Direction
    Vector2 xi(GPUDistribution::Uniform<float>(rng),
               GPUDistribution::Uniform<float>(rng));
    Vector3 direction = HemiDistribution::HemiCosineCDF(xi, pdf);
    direction.NormalizeSelf();

    // Cos Tetha
    float nDotL = max(normal.Dot(direction), 0.0f);

    // Ray out
    Vector3 gNnormalW = GPUSurface::NormalWorld(surface.worldToTangent);
    Vector3 outPos = position + gNnormalW * MathConstants::Epsilon;
    Vector3 outDir = GPUSurface::ToWorld(direction, surface.worldToTangent);

    // Ray out
    wo = RayF(outDir, outPos);

    // Radiance Calculation
    const Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
    return nDotL * albedo * MathConstants::InvPi;
}

__device__ __forceinline__
float LambertPDF(const Vector3& wo,
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
        normal = (*matData.dNormal[matId])(surface.uv);       
    normal = GPUSurface::NormalWorld(surface.worldToTangent);

    float pdf = max(wo.Dot(normal), 0.0f);
    pdf *= MathConstants::InvPi;
    return pdf;
}

__device__ __forceinline__
Vector3 LambertEvaluate(// Input
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
        normal = (*matData.dNormal[matId])(surface.uv);

    // Calculate lightning in world space since
    // wo is already in world space
    normal = GPUSurface::ToWorld(normal, surface.worldToTangent);

    float nDotL = max(normal.Dot(wo), 0.0f);
    const Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
    return nDotL * albedo * MathConstants::InvPi;
}