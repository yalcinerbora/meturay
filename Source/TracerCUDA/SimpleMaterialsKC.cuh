#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"
#include "RayLib/CudaCheck.h"

#include "Random.cuh"
#include "ImageFunctions.cuh"
#include "MaterialFunctions.cuh"
#include "TracerFunctions.cuh"
#include "GPUSurface.h"

__device__ __forceinline__
Vector3 LambertCSample(// Sampled Output
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
                       const AlbedoMatData& matData,
                       const HitKey::Type& matId,
                       uint32_t sampleIndex)
{
    // No medium change
    outMedium = &m;
    // Ray Selection
    const Vector3& position = pos;
    const Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
    // Generate New Ray Directiion
    Vector2 xi(GPUDistribution::Uniform<float>(rng),
               GPUDistribution::Uniform<float>(rng));
    Vector3 direction = HemiDistribution::HemiCosineCDF(xi, pdf);
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
    return nDotL * matData.dAlbedo[matId] * MathConstants::InvPi;
}

__device__ __forceinline__
float LambertCPdf(const Vector3& wo,
                  const Vector3& wi,
                  const Vector3& pos,
                  const GPUMediumI& m,
                  //
                  const BasicSurface& surface,
                  // Constants
                  const AlbedoMatData& matData,
                  const HitKey::Type& matId)
{
    const Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
    float pdf = max(wo.Dot(normal), 0.0f);

    pdf *= MathConstants::InvPi;
    return pdf;
}

__device__ __forceinline__
Vector3 LambertCEvaluate(// Input
                         const Vector3& wo,
                         const Vector3& wi,
                         const Vector3& pos,
                         const GPUMediumI& m,
                         //
                         const BasicSurface& surface,
                         // Constants
                         const AlbedoMatData& matData,
                         const HitKey::Type& matId)
{
    Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
    float nDotL = max(normal.Dot(wo), 0.0f);
    return nDotL * matData.dAlbedo[matId] * MathConstants::InvPi;
}

__device__ __forceinline__
Vector3 ReflectSample(// Sampled Output
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
                      const ReflectMatData& matData,
                      const HitKey::Type& matId,
                      uint32_t sampleIndex)
{
    // No medium change
    outMedium = &m;

    // Fetch Mat
    Vector4 data = matData.dAlbedoAndRoughness[matId];
    Vector3 albedo = data;
    float roughness = data[3];

    const Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
    // Calculate Reflection
    if(roughness == 0.0f)
    {
        // Singularity Just Reflect
        wo = RayF(wi, pos).Reflect(normal);
        wo.AdvanceSelf(MathConstants::Epsilon, normal);

        pdf = 1.0f;
        return albedo;
    }
    else
    {
        // TODO: Do a delta distribution towards reflection
        return Zero3;
    }
}

__device__ __forceinline__
Vector3 ReflectEvaluate(// Input
                        const Vector3& wo,
                        const Vector3& wi,
                        const Vector3& pos,
                        const GPUMediumI& m,
                        //
                        const BasicSurface& surface,
                        // Constants
                        const ReflectMatData& matData,
                        const HitKey::Type& matId)
{
    // You cant sample perfect material from an arbitrary location
    return Zero3;
}

__device__ __forceinline__
Vector3 RefractSample(// Sampled Output
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
                      const RefractMatData& matData,
                      const HitKey::Type& matId,
                      uint32_t sampleIndex)
{
    // Fetch Mat
    Vector3 albedo = matData.dAlbedo[matId];
    uint32_t mediumIndex = matData.mediumIndices[matId];
    float iIOR = matData.dMediums[mediumIndex]->IOR();
    float dIOR = matData.dMediums[matData.baseMediumIndex]->IOR();

    const Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
    const Vector3& position = pos;

    // Check if we are exiting or entering
    float nDotI = wi.Dot(normal);
    bool entering = (nDotI >= 0.0f);

    // Determine medium index of refractions
    float fromMedium = m.IOR();
    float toMedium = (entering) ? iIOR : dIOR;

    // Normal also needs to be on the same side of the surface for the funcs
    // to work
    Vector3 refNormal = (entering) ? normal : (-normal);

    // Calculate Frenel Term
    float f = TracerFunctions::FrenelDielectric(abs(nDotI), fromMedium, toMedium);

    // Sample ray according to the frenel term
    float xi = GPUDistribution::Uniform<float>(rng);
    if(xi < f)
    {
        // RNG choose to sample Reflection case
        wo = RayF(wi, position).Reflect(refNormal);
        wo.AdvanceSelf(MathConstants::Epsilon, refNormal);
        // Frenel term is used to sample thus pdf is f
        pdf = f;
        // We reflected off of surface no medium change
        outMedium = &m;

        float nDotL = wo.getDirection().Dot(refNormal);
        return f * albedo;
    }
    else
    {
        // Write new medium
        uint32_t outMediumIndex = (entering) ? mediumIndex : matData.baseMediumIndex;
        outMedium = matData.dMediums[outMediumIndex];
        // Write pdf
        pdf = 1.0f - f;

        // Refraction is choosen
        // Convert wi, refract func needs
        // the direction to be towards surface
        RayF rayIn(wi, position);
        // Get refracted ray
        bool refracted = rayIn.Refract(wo, refNormal, fromMedium, toMedium);
        // Since Frenel term is used to sample,
        // code should not arrive here (raise exception)
        // Update:
        // Well code does arrive here rarely (due to numerical error i guess)
        // so return zero instead :)
        if(!refracted)
        {
            KERNEL_DEBUG_LOG("CUDA Error: RefractMat reflected!\n");
            pdf = 0.0f;
            return Zero3;
            //__threadfence();
            //__trap();
        }

        // We passed the boundary
        // advance towards opposite direction
        wo.AdvanceSelf(MathConstants::Epsilon, -refNormal);

        // Factor in the radiance discrapency due to refraction
        // Medium change causes rays to be scatter/focus
        // Since we try to calculate radiance towards that ray
        float radianceChangeFactor = (fromMedium * fromMedium) / (toMedium * toMedium);
        // Final Factor
        return albedo * radianceChangeFactor * (1.0f - f);
    }
}

__device__ __forceinline__
Vector3 RefractEvaluate(// Input
                        const Vector3& wo,
                        const Vector3& wi,
                        const Vector3& pos,
                        const GPUMediumI& m,
                        //
                        const BasicSurface& surface,
                        // Constants
                        const RefractMatData& matData,
                        const HitKey::Type& matId)
{
    return Zero3;
}