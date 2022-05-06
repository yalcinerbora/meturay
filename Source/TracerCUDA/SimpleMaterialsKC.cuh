#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"
#include "RayLib/CudaCheck.h"

#include "RNGenerator.h"
#include "ImageFunctions.cuh"
#include "MaterialFunctions.h"
#include "MetaMaterialFunctions.cuh"
#include "TracerFunctions.cuh"
#include "GPUSurface.h"

struct LambertConstFuncs
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
                   const AlbedoMatData& matData,
                   const HitKey::Type& matId,
                   uint32_t sampleIndex)
    {
        // No medium change
        outMedium = &m;
        // Ray Selection
        const Vector3& position = pos;
        const Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
        // Generate New Ray Direction
        Vector2 xi(rng.Uniform(), rng.Uniform());
        Vector3 direction = HemiDistribution::HemiCosineCDF(xi, pdf);
        direction.NormalizeSelf();

        // Generated direction vector is on surface space (hemispherical)
        // Convert it to normal oriented hemisphere (world space)
        QuatF q = Quat::RotationBetweenZAxis(normal);
        direction = q.ApplyRotation(direction);

        // Cos Theta
        float nDotL = max(normal.Dot(direction), 0.0f);

        // Ray out
        wo = RayF(direction, position);
        // BSDF Calculation
        return nDotL * matData.dAlbedo[matId] * MathConstants::InvPi;
    }

    __device__ inline static
    float Pdf(const Vector3& wo,
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

    __device__ inline static
    Vector3 Evaluate(// Input
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

    static constexpr auto& Emit         = EmitEmpty<AlbedoMatData, BasicSurface>;
    static constexpr auto& IsEmissive   = IsEmissiveFalse<AlbedoMatData>;
    static constexpr auto& Specularity  = SpecularityDiffuse<AlbedoMatData, BasicSurface>;
};

struct ReflectMatFuncs
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
            wo.NudgeSelf(surface.WorldGeoNormal());
            pdf = 1.0f;
            return albedo;
        }
        else
        {
            // TODO: Do a delta distribution towards reflection
            return Zero3;
        }
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
                     const ReflectMatData& matData,
                     const HitKey::Type& matId)
    {
        // You cant evaluate perfect material from an arbitrary location
        return Zero3;
    }

    static constexpr auto& Pdf          = PdfZero<ReflectMatData, BasicSurface>;
    static constexpr auto& Emit         = EmitEmpty<ReflectMatData, BasicSurface>;
    static constexpr auto& IsEmissive   = IsEmissiveFalse<ReflectMatData>;
    static constexpr auto& Specularity  = SpecularityPerfect<ReflectMatData, BasicSurface>;
};

struct RefractMatFuncs
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
        bool entering = !surface.backSide;

        // Determine medium index of refractions
        float fromMedium = m.IOR();
        float toMedium = (entering) ? iIOR : dIOR;

        // Surface is aligned with the ray (N dot Dir is always positive)
        Vector3 refNormal = normal;
        // Calculate Fresnel Term
        float f = TracerFunctions::FrenelDielectric(fabs(wi.Dot(normal)),
                                                    fromMedium, toMedium);

        //printf("[%s] P:(%f, %f, %f) Wi:(%f, %f, %f) N:(%f, %f, %f) - F %f, From %f, To %f\n",
        //       entering ? "entering" : "exiting",
        //       pos[0], pos[1], pos[2],
        //       wi[0], wi[1], wi[2],
        //       normal[0], normal[1], normal[2],
        //       f, fromMedium, toMedium);

        // Sample ray according to the Fresnel term
        float xi = rng.Uniform();
        if(xi < f)
        {
            // RNG choose to sample Reflection case
            wo = RayF(wi, position).Reflect(refNormal);
            wo.NudgeSelf(surface.WorldGeoNormal());
            // Fresnel term is used to sample thus pdf is f
            pdf = f;
            // We reflected off of surface no medium change
            outMedium = &m;

            //printf("Reflected no medium change Wo[%f, %f, %f]\n",
            //       wo.getDirection()[0], wo.getDirection()[1], wo.getDirection()[2]);

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

            //printf("Refracted new medium IOR %f, Wo[%f, %f, %f]\n", outMedium->IOR(),
            //       wo.getDirection()[0], wo.getDirection()[1], wo.getDirection()[2]);

            // Refraction is chosen
            // Convert wi, refract function needs
            // the direction to be towards surface
            RayF rayIn(wi, position);
            // Get refracted ray
            bool refracted = rayIn.Refract(wo, refNormal, fromMedium, toMedium);
            // Since Fresnel term is used to sample,
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
            wo.NudgeSelf(-surface.WorldGeoNormal());

            // Factor in the radiance discrepancy due to refraction
            // Medium change causes rays to be scatter/focus
            // Since we try to calculate radiance towards that ray
            //float radianceChangeFactor = (fromMedium * fromMedium) / (toMedium * toMedium);
            float radianceChangeFactor = 1.0f;
            // Final Factor
            return albedo * radianceChangeFactor * (1.0f - f);
        }
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
                     const RefractMatData& matData,
                     const HitKey::Type& matId)
    {
        return Zero3;
    }

    static constexpr auto& Pdf          = PdfZero<RefractMatData, BasicSurface>;
    static constexpr auto& Emit         = EmitEmpty<RefractMatData, BasicSurface>;
    static constexpr auto& IsEmissive   = IsEmissiveFalse<RefractMatData>;
    static constexpr auto& Specularity  = SpecularityPerfect<RefractMatData, BasicSurface>;
};