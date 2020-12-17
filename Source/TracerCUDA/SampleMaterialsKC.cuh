#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"

#include "Random.cuh"
#include "TextureStructs.h"
#include "ImageFunctions.cuh"
#include "MaterialFunctions.cuh"
#include "TracerFunctions.cuh"
#include "GPUSurface.h"

__device__ inline
Vector3 EmitConstant(// Input
                     const Vector3& wo,
                     const Vector3& pos,
                     const GPUMediumI& m,
                     //
                     const EmptySurface& surface,
                     const TexCoords* uvs,
                     // Constants
                     const EmissiveMatData& matData,
                     const HitKey::Type& matId)
{
    return matData.dAlbedo[matId];
}

__device__ inline
Vector3 LambertSample(// Sampled Output
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
                      const AlbedoMatData& matData,
                      const HitKey::Type& matId,
                      uint32_t sampleIndex)
{
    // No medium change
    outMedium = &m;

    // Ray Selection
    const Vector3& position = pos;
    const Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
    //const Vector3& normal = surface.normal;
    // Generate New Ray Directiion
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
    return nDotL * matData.dAlbedo[matId] * MathConstants::InvPi;
}

__device__ inline
Vector3 LambertEvaluate(// Input
                        const Vector3& wo,
                        const Vector3& wi,
                        const Vector3& pos,
                        const GPUMediumI& m,
                        //
                        const BasicSurface& surface,
                        const TexCoords* uvs,
                        // Constants
                        const AlbedoMatData& matData,
                        const HitKey::Type& matId)
{
    //const Vector3& normal = surface.normal;
    Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent).Normalize();
    
    //float nDotL = max(normal.Dot(wo), 0.0f);
    //Vector3 wOTangentSpace = GPUSurface::ToTangent(wo, surface.worldToTangent);
    //float nDotL = max(wOTangentSpace[2], 0.0f);
    
    //return nDotL * matData.dAlbedo[matId] * MathConstants::InvPi;
    return normal;
}

__device__ inline
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
                      const TexCoords* uvs,
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

__device__ inline
Vector3 ReflectEvaluate(// Input
                        const Vector3& wo,
                        const Vector3& wi,
                        const Vector3& pos,
                        const GPUMediumI& m,
                        //
                        const BasicSurface& surface,
                        const TexCoords* uvs,
                        // Constants
                        const ReflectMatData& matData,
                        const HitKey::Type& matId)
{
    // You cant sample perfect material from an arbitrary location
    return Zero3;
}

__device__ inline
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
                      const TexCoords* uvs,
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

    if(!(f <= 1.0f && f >= 0.0f))
    {
        printf("frenel %f\n", f);
    }

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
        //pdf = (1.0f - f);
        //return Zero3;

        // Refraction is choosen
        // Convert wi, refract func needs 
        // the direction to be towards surface    
        RayF rayIn(wi, position);
        // Get refracted ray
        bool refracted = rayIn.Refract(wo, refNormal, fromMedium, toMedium);
        // Since Frenel term is used to sample,
        // code should not arrive here (raise exception)
        if(!refracted)
        {
            printf("CUDA Fatal Error: RefractMat reflected!\n");
            return Zero3;
            //__threadfence();
            //__trap(); 
        }

        // We passed the boundary 
        // advance towards opposite direction
        wo.AdvanceSelf(MathConstants::Epsilon, -refNormal);
        
        pdf = 1.0f - f;

        // Change medium
        uint32_t outMediumIndex = (entering) ? mediumIndex : matData.baseMediumIndex;
        outMedium = matData.dMediums[outMediumIndex];

        return albedo * (1.0f - f);
    }
}

__device__ inline
Vector3 RefractEvaluate(// Input
                        const Vector3& wo,
                        const Vector3& wi,
                        const Vector3& pos,
                        const GPUMediumI& m,
                        //
                        const BasicSurface& surface,
                        const TexCoords* uvs,
                        // Constants
                        const RefractMatData& matData,
                        const HitKey::Type& matId)
{
    return Zero3;
}