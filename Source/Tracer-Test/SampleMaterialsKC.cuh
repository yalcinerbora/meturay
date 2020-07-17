#pragma once

#include "RayAuxStruct.h"
#include "MaterialDataStructs.h"
#include "SurfaceStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"

#include "TracerLib/Random.cuh"
#include "TracerLib/TextureStructs.h"
#include "TracerLib/ImageFunctions.cuh"
#include "TracerLib/MaterialFunctions.cuh"

__device__ inline
Vector3 LambertSample(// Sampled Output
                      RayF& wo,
                      float& pdf,
                      // Input
                      const Vector3& wi,
                      const Vector3& pos,
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
    Vector3 direction = HemiDistribution::HemiCosineCDF(xi);
    direction.NormalizeSelf();

    // Direction vector is on surface space (hemisperical)
    // Convert it to normal oriented hemisphere (world space)
    QuatF q = QuatF::RotationBetweenZAxis(normal);
    direction = q.ApplyRotation(direction);

    // Cos Tetha
    float nDotL = max(normal.Dot(direction), 0.0f);
    // Pdf
    pdf = nDotL * MathConstants::InvPi;
    // Ray out
    wo = RayF(direction, position);

    // Illumination Calculation
    return nDotL * matData.dAlbedo[matId] * MathConstants::InvPi;
}

__device__ inline
Vector3 LambertEvaluate(// Input
                         const Vector3& wo,
                         const Vector3& wi,
                         const Vector3& pos,
                         const BasicSurface& surface,
                         const TexCoords* uvs,
                         // Constants
                         const AlbedoMatData& matData,
                         const HitKey::Type& matId)
{
    // Lambert mat is constant throught
    return matData.dAlbedo[matId] * MathConstants::InvPi;
}

__device__ inline
Vector3 ReflectSample(// Sampled Output
                      RayF& wo,
                      float& pdf,
                      // Input
                      const Vector3& wi,
                      const Vector3& pos,
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
    }
}

__device__ inline
Vector3 ReflectEvaluate(// Input
                        const Vector3& wo,
                        const Vector3& wi,
                        const Vector3& pos,
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
                      // Input
                      const Vector3& wi,
                      const Vector3& pos,
                      const BasicSurface& surface,
                      const TexCoords* uvs,
                      MediumBoundary boundary,
                      // I-O
                      RandomGPU& rng,
                      // Constants
                      const RefractMatData& matData,
                      const HitKey::Type& matId,
                      uint32_t sampleIndex)
{
    // Fetch Mat
    Vector4 data = matData.dAlbedoAndIndex[matId];
    Vector3 albedo = data;
    float index = data[3];

    const Vector3& normal = surface.normal;
    const Vector3& position = pos;

    // Check if we are exiting or entering
    float orientationFactor = wi.Dot(normal);
    bool entering = (orientationFactor <= 0.0f);

    // Determine medium index of refractions
    float fromMedium = boundary.fromIOR;
    float toMedium = (entering) ? index : boundary.toIOR;
    assert(entering && (__half2float(boundary.toIOR) == index));

    // Calculate Frenel Term
    float f;

    // Sample ray according to the frenel term
    float xi = GPUDistribution::Uniform<float>(rng);
    if(xi < f)
    {
        // RNG choose to sample Reflection case
        wo = RayF(wi, position).Reflect(normal);
        pdf = 1.0f;

        // Frenel term is used to sample thus pdf is f
        pdf = f;
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
        if(!refracted) __trap();

        // Frenel term is used to sample thus pdf is (1-f)
        pdf = 1.0f - f;
    }

    // return radiance factor
    return albedo;


    // Check if frenel effect is in place
    if(fromMedium < toMedium)
    {
        // There
    }
    else
    {
        // No frenel
    }
}

__device__ inline
Vector3 RefractEvaluate(// Input
                        const Vector3& wo,
                        const Vector3& wi,
                        const Vector3& pos,
                        const BasicSurface& surface,
                        const TexCoords* uvs,
                        // Constants
                        const RefractMatData& matData,
                        const HitKey::Type& matId)
{
    return Zero3;
}

//inline void BasicReflectPTShade(// Output
//                                ImageGMem<Vector4f> gImage,
//                                //
//                                HitKey* gBoundaryMat,
//                                RayGMem* gOutRays,
//                                RayAuxBasic* gOutRayAux,
//                                const uint32_t maxOutRay,
//                                // Input as registers
//                                const RayReg& ray,
//                                const BasicSurface& surface,
//                                const RayAuxBasic& aux,
//                                // RNG
//                                RandomGPU& rng,
//                                // 
//                                const BasicEstimatorData& estData,
//                                // Input as global memory
//                                const ReflectMatData& gMatData,
//                                const HitKey::Type& matId)
//{
//    assert(maxOutRay == REFLECTPT_MAX_OUT_RAY);
//    // Inputs
//    const RayAuxBasic& auxIn = aux;
//    const RayReg rayIn = ray;
//    // Outputs
//    RayReg rayOut = EMPTY_RAY_REGISTER;
//    RayAuxBasic auxOut = auxIn;
//    auxOut.depth++;
//    // Do not change the ray type
//
//    // Skip if light ray
//    if(auxIn.type == RayType::NEE_RAY)
//    {
//        // Generate Dummy Ray and Terminate
//        rayOut.Update(gOutRays, 0);
//        gBoundaryMat[0] = HitKey::InvalidKey;
//        return;
//    }
//
//    // Fetch Mat
//    Vector4 matData = gMatData.dAlbedoAndRoughness[matId];
//    Vector3 albedo = matData;
//    float roughness = matData[3];
//
//    // Fetch Data
//    const Vector3 position = rayIn.ray.AdvancedPos(rayIn.tMax);
//    const Vector3 direction = rayIn.ray.getDirection();
//    const Vector3 normal = surface.normal;
//
//    // Russian Roulette
//    float avgThroughput = auxOut.radianceFactor.Dot(Vector3f(0.333f));
//    // float avgThroughput = gMatData.dAlbedo[matId].Dot(Vector3f(0.333f));
//
//    if(auxIn.depth <= 3 &&
//       !GPUEventEstimatorBasic::TerminatorFunc(auxOut.radianceFactor,
//                                               avgThroughput,
//                                               rng))
//    {
//         // Calculate Reflection   
//        if(roughness == 1.0f)
//        {
//            // Singularity Just Reflect
//            rayOut.ray = RayF(direction, position).Reflect(normal);
//            rayOut.ray.AdvanceSelf(MathConstants::Epsilon);
//            rayOut.tMin = 0.0f;
//            rayOut.tMax = INFINITY;
//
//            auxOut.radianceFactor *= albedo;
//            gOutRayAux[0] = auxOut;
//        }
//        else
//        {
//            // TODO:
//            // Use a normal distribution for sampling wrt. roughness
//        }
//    }
//
//    gBoundaryMat[0] = HitKey::InvalidKey;
//    rayOut.Update(gOutRays, 0);
//}
//
//__device__
//inline void BasicRefractPTShade(// Output
//                                ImageGMem<Vector4f> gImage,
//                                //
//                                HitKey* gBoundaryMat,
//                                RayGMem* gOutRays,
//                                RayAuxBasic* gOutRayAux,
//                                const uint32_t maxOutRay,
//                                // Input as registers
//                                const RayReg& ray,
//                                const BasicSurface& surface,
//                                const RayAuxBasic& aux,
//                                // RNG
//                                RandomGPU& rng,
//                                // 
//                                const BasicEstimatorData& estData,
//                                // Input as global memory
//                                const RefractMatData& gMatData,
//                                const HitKey::Type& matId)
//{
//    assert(maxOutRay == REFRACTPT_MAX_OUT_RAY);
//
//    // Inputs
//    const RayAuxBasic& auxIn = aux;
//    const RayReg rayIn = ray;
//    // Outputs
//    RayReg rayOut = EMPTY_RAY_REGISTER;
//    RayAuxBasic auxOut = auxIn;
//    auxOut.depth++;
//    // Do not change the ray type
//
//    // Skip if light ray
//    if(auxIn.type == RayType::NEE_RAY)
//    {
//        // Generate Dummy Ray and Terminate
//        rayOut.Update(gOutRays, 0);
//        gBoundaryMat[0] = HitKey::InvalidKey;
//        return;
//    }
//
//    // Fetch Mat
//    Vector4 matData = gMatData.dAlbedoAndIndex[matId];
//    Vector3 albedo = matData;
//    float index = matData[3];
//
//    // Fetch Data
//    const Vector3 position = rayIn.ray.AdvancedPos(rayIn.tMax);
//    const Vector3 direction = rayIn.ray.getDirection();
//    const Vector3 normal = surface.normal;
//
//    // Russian Roulette
//    float avgThroughput = auxOut.radianceFactor.Dot(Vector3f(0.333f));
//    // float avgThroughput = gMatData.dAlbedo[matId].Dot(Vector3f(0.333f));
//
//    if(auxIn.depth <= 3 &&
//       !GPUEventEstimatorBasic::TerminatorFunc(auxOut.radianceFactor,
//                                               avgThroughput,
//                                               rng))
//    {
//        // Check if we are exiting or entering
//        bool entering = (__half2float(auxIn.mediumIndex) == 1.0f);
//        float fromMedium = auxIn.mediumIndex;
//        float toMedium = (entering) ? index : 1.0f;
//        
//        // Utilize frenel term to select reflection or refraction
//        RayF r;
//        bool refracted = ray.ray.Refract(r, normal, fromMedium, toMedium);
//
//        if(entering)
//        {
//
//        }
//        else
//        {
//            // No frenel
//        }
//    }
//
//    gBoundaryMat[0] = HitKey::InvalidKey;
//    rayOut.Update(gOutRays, 0);
//}