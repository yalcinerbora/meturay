#pragma once

#include "RayAuxStruct.h"
#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"

#include "TracerLib/Random.cuh"
#include "TracerLib/TextureStructs.h"
#include "TracerLib/ImageFunctions.cuh"
#include "TracerLib/MaterialFunctions.cuh"
#include "TracerLib/TracerFunctions.cuh"
#include "TracerLib/SurfaceStructs.h"

template <class Surface>
__device__ inline
Vector3 EmitConstant(// Input
                     const Vector3& wo,
                     const Vector3& pos,
                     const GPUMedium& m,
                     //
                     const Surface& surface,
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
                      GPUMedium& outMedium,
                      // Input
                      const Vector3& wi,
                      const Vector3& pos,
                      const GPUMedium& m,
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
    // Ray Selection
    const Vector3& position = pos;
    const Vector3& normal = surface.normal;
    // Generate New Ray Directiion
    Vector2 xi(GPUDistribution::Uniform<float>(rng),
               GPUDistribution::Uniform<float>(rng));
    Vector3 direction = HemiDistribution::HemiCosineCDF(xi, pdf);
    //Vector3 direction = HemiDistribution::HemiUniformCDF(xi, pdf);
    direction.NormalizeSelf();

    // Direction vector is on surface space (hemisperical)
    // Convert it to normal oriented hemisphere (world space)
    QuatF q = Quat::RotationBetweenZAxis(normal);
    direction = q.ApplyRotation(direction);

    // Cos Tetha
    float nDotL = max(normal.Dot(direction), 0.0f);

    //pdf = nDotL * MathConstants::InvPi;

    // Ray out
    Vector3 outPos = position + normal * MathConstants::Epsilon;
    wo = RayF(direction, outPos);
    // Illumination Calculation
    return nDotL * matData.dAlbedo[matId] * MathConstants::InvPi;
}

__device__ inline
Vector3 LambertEvaluate(// Input
                        const Vector3& wo,
                        const Vector3& wi,
                        const Vector3& pos,
                        const GPUMedium& m,
                        //
                        const BasicSurface& surface,
                        const TexCoords* uvs,
                        // Constants
                        const AlbedoMatData& matData,
                        const HitKey::Type& matId)
{
    const Vector3& normal = surface.normal;
    // Cos Tetha
    float nDotL = max(normal.Dot(wo), 0.0f);
    return nDotL * matData.dAlbedo[matId] * MathConstants::InvPi;
}

__device__ inline
Vector3 ReflectSample(// Sampled Output
                      RayF& wo,
                      float& pdf,
                      GPUMedium& outMedium,
                      // Input
                      const Vector3& wi,
                      const Vector3& pos,
                      const GPUMedium& m,
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
    // Fetch Mat
    Vector4 data = matData.dAlbedoAndRoughness[matId];
    Vector3 albedo = data;
    float roughness = data[3];

    const Vector3& normal = surface.normal;
    const Vector3& position = pos;

    // No medium change
    outMedium = m;

    // Calculate Reflection   
    if(roughness == 1.0f)
    {
        // Singularity Just Reflect
        wo = RayF(wi, position).Reflect(normal);
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
                        const GPUMedium& m,
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
                      GPUMedium& outMedium,
                      // Input
                      const Vector3& wi,
                      const Vector3& pos,
                      const GPUMedium& m,
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
    float iOIR = matData.dMedium[matId].IOR();

    const Vector3& normal = surface.normal;
    const Vector3& position = pos;

    // Check if we are exiting or entering
    float nDotL = wi.Dot(normal);
    bool entering = (nDotL <= 0.0f);

    // Determine medium index of refractions
    float fromMedium = m.IOR();
    float toMedium = (entering) ? iOIR : 1.0f;
    
    // Calculate Frenel Term
    float f = TracerFunctions::FrenelDielectric(abs(nDotL), fromMedium, toMedium);

    // Sample ray according to the frenel term
    float xi = GPUDistribution::Uniform<float>(rng);
    if(xi < f)
    {
        // RNG choose to sample Reflection case
        wo = RayF(wi, position).Reflect(normal);
        // Frenel term is used to sample thus pdf is f
        pdf = f;
        // We reflected off of surface no medium change
        outMedium = m;
    }
    else
    {
        // Refraction is choosen to sample
        // Convert wi, refract func needs 
        // the direction to be towards surface    
        RayF rayIn(-wi, position);
        // Normal also needs to be on the same side of the surface for the func
        Vector3 refNormal = (entering) ? normal : (-normal);
        // Get refracted ray
        bool refracted = rayIn.Refract(wo, refNormal, fromMedium, toMedium);
        // Since Frenel term is used to sample
        // code should not arrive here (raise exception)
        if(!refracted)
        {
            __threadfence();
            __trap(); 
        }

        // Return medium
        outMedium = (entering) ? (matData.dMedium[matId]) 
                               : (*matData.dDefaultMedium);

        // Frenel term is used to sample thus pdf is (1-f)
        pdf = 1.0f - f;
    }
    // return radiance factor
    return albedo;
}

__device__ inline
Vector3 RefractEvaluate(// Input
                        const Vector3& wo,
                        const Vector3& wi,
                        const Vector3& pos,
                        const GPUMedium& m,
                        //
                        const BasicSurface& surface,
                        const TexCoords* uvs,
                        // Constants
                        const RefractMatData& matData,
                        const HitKey::Type& matId)
{
    return Zero3;
}