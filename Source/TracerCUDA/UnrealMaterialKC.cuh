#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"

#include "RNGenerator.h"
#include "ImageFunctions.cuh"
#include "MaterialFunctions.h"
#include "TracerFunctions.cuh"
#include "GPUSurface.h"
#include "TracerConstants.h"
#include "MetaMaterialFunctions.cuh"

__device__ inline
Vector3 CalculateF0(const Vector3f& baseAlbedo, float metallic, float specular)
{
    // https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
    // Utilizing proposed specular value to blend base color with specular parameter

    static constexpr float SpecularMax = 0.08f;
    specular *= SpecularMax;

    return Vector3f::Lerp(Vector3f(specular), baseAlbedo, metallic);
}

struct UnrealDeviceFuncs
{
    __device__ inline static
    float Specularity(const UVSurface& surface,
                      const UnrealMatData& matData,
                      const HitKey::Type& matId)
    {
        float roughness = (*matData.dRoughness[matId])(surface.uv);
        //float alpha = roughness * roughness;
        //return 1.0f - alpha;
        return 1.0f - roughness;
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
                   const UVSurface& surface,
                   // I-O
                   RNGeneratorGPUI& rng,
                   // Constants
                   const UnrealMatData& matData,
                   const HitKey::Type& matId,
                   uint32_t sampleIndex)
    {
        // No medium change
        outMedium = &m;

        // Acquire Parameters
        // Check if normal mapping is present
        float roughness = (*matData.dRoughness[matId])(surface.uv);
        float metallic = (*matData.dMetallic[matId])(surface.uv);
        float specular = (*matData.dSpecular[matId])(surface.uv);
        Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
        float alpha = roughness * roughness;
        // Since we using capital terms alias wi as V
        Vector3 V = GPUSurface::ToTangent(wi, surface.worldToTangent);
        // Compute tangent space if normal map is present
        Vector3 N = ZAxis;
        if(matData.dNormal[matId])
        {
            N = (*matData.dNormal[matId])(surface.uv).Normalize();
            // Align the half vector with the normal map normal
            QuatF normalRot = Quat::RotationBetweenZAxis(N);
            V = normalRot.ApplyRotation(V);
        }

        // Sample a H (Half Vector)
        Vector3 H = TracerFunctions::VNDFGGXSmithSample(pdf, V, alpha, rng);
        // Reflect the light using microfacet normal
        Vector3 L = 2.0f * V.Dot(H) * H - V;

        // Pre-check the shadowing term switches
        float LdH = max(L.Dot(H), 0.0f);
        float VdH = max(V.Dot(H), 0.0f);
        if(LdH == 0.0f || // wi is masked
           VdH == 0.0f || // wo is shadowed
           pdf == 0.0f)
        {
            pdf = 0.0f;
            return Zero3f;
        }

        // VNDFGGXSmithSample returns sampling of H Vector
        // convert it to sampling probability of L Vector
        pdf /= (4.0f * LdH);

        // Now do the BRDF
        // Normal Distribution Function (GGX)
        float D = TracerFunctions::DGGX(N.Dot(H), alpha);
        // Shadowing Term (Smith Model)
        float G = TracerFunctions::GSmithCorralated(V, L, alpha);
        // Fresnel Term (Schlick's Approx)
        Vector3f f0 = CalculateF0(albedo, metallic, specular);
        Vector3f F = TracerFunctions::FSchlick(VdH, f0);
        //=======================//
        // Calculate Reflectance //
        //=======================//
        // Blend between albedo-black for metallic material
        float NdL = max(N.Dot(L), 0.0f);
        Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
        Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;
        // Notice that NdL terms are canceled out
        float NdV = max(N.Dot(V), 0.0f);
        Vector3f specularTerm = D * F * G * 0.25f / NdV;
        specularTerm = (NdV == 0.0f) ? Zero3 : specularTerm;

        //=======================//
        //  Calculate Direction  //
        //=======================//
        // We need to slightly nudge the ray start
        // to prevent self intersection
        // Normal is on tangent space so convert it to world space
        // Convert Normal to World Space
        wo = RayF(GPUSurface::ToSpace(L, surface.worldToTangent), pos);
        wo.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);
        // PDF is already written
        // Finally return Radiance
        // All Done!
        return /*diffuseTerm +*/ specularTerm;





        //// Gen L (aka. wo)
        //Vector3 L = 2.0f * V.Dot(H) * H - V;

        //// BRDF Calculation
        ////float NdL = max(N.Dot(L), 0.0f);
        ////float NdV = max(N.Dot(V), 0.0f);
        ////float NdH = max(N.Dot(H), 0.0f);
        ////float VdH = max(V.Dot(H), 0.0f);
        ////float LdH = max(L.Dot(H), 0.0f);
        //float NdL = abs(N.Dot(L));
        //float NdV = abs(N.Dot(V));
        //float NdH = abs(N.Dot(H));
        //float VdH = abs(V.Dot(H));
        //float LdH = abs(L.Dot(H));

        //// Edge cases
        //if(NdV == 0.0f || LdH == 0.0f)
        //{
        //    pdf = 0.0f;
        //    return Zero3;
        //}

        //// GGXSample returns sampling of H Vector
        //// convert it to sampling probability of L Vector
        //pdf /= (4.0f * (LdH));

        //// If material is nearly perfect specular,
        //// set pdf to 1 (convert this material to a mirror).
        //// Note that this is true only for sampling, evaluate and pdf will return
        //// zero (since you can't find perfect reflection of the view vector by chance)
        //using namespace TracerConstants;
        //bool isSpecular = (Specularity(surface, matData, matId) >= SPECULAR_THRESHOLD);
        //if(isSpecular)
        //{
        //    pdf = 1.0f;
        //    D = 1.0f;
        //}

        //// Shadowing Term (Schlick Model)
        ////float G = TracerFunctions::GSchlick(NdL, roughness) *
        ////          TracerFunctions::GSchlick(NdV, roughness);
        //float G = 1.0f;
        //// Fresnel Term (Schlick's Approx)
        //Vector3f f0 = CalculateF0(albedo, metallic, specular);
        ////Vector3f F = (isSpecular) ? f0 : TracerFunctions::FSchlick(VdH, f0);
        //Vector3f F = Vector3f(1.0f);

        ////=======================//
        //// Calculate Reflectance //
        ////=======================//
        //// Blend between albedo-black for metallic material
        //Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
        //Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;
        //// Notice that NdL terms are canceled out
        //Vector3f specularTerm = D * F * G * 0.25f / NdV;
        //specularTerm = (NdV == 0.0f) ? Zero3 : specularTerm;

        ////=======================//
        ////  Calculate Direction  //
        ////=======================//
        //// We need to slightly nudge the ray start
        //// to prevent self intersection
        //// Normal is on tangent space so convert it to world space
        //// Convert Normal to World Space
        //wo = RayF(GPUSurface::ToSpace(L, surface.worldToTangent), pos);
        //wo.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);
        //// PDF is already written
        //// Finally return Radiance
        //// All Done!
        //return /*diffuseTerm +*/ specularTerm;
    }

    __device__ inline static
    float Pdf(// Input
              const Vector3& wo,
              const Vector3& wi,
              const Vector3& pos,
              const GPUMediumI& m,
              //
              const UVSurface& surface,
              // Constants
              const UnrealMatData& matData,
              const HitKey::Type& matId)
    {
        // It is impossible to find exact wo <=> wi
        // with a chance
        using namespace TracerConstants;
        bool isSpecular = (Specularity(surface, matData, matId) >= SPECULAR_THRESHOLD);
        if(isSpecular)
        {
            return 0.0f;
        }

        float roughness = (*matData.dRoughness[matId])(surface.uv);
        Vector3 N = ZAxis;
        if(matData.dNormal[matId])
            N = (*matData.dNormal[matId])(surface.uv).Normalize();

        Vector3 L = GPUSurface::ToTangent(wo, surface.worldToTangent);
        Vector3 V = GPUSurface::ToTangent(wi, surface.worldToTangent);
        Vector3 H = (L + V).Normalize();

        float NdH = abs(N.Dot(H));

        //float sinTheta = sqrt(max(0.0f, 1.0f - NdH * NdH));
        float pdf = TracerFunctions::DGGX(NdH, roughness) * NdH;

        // GGXSample returns sampling of H Vector
        // convert it to sampling probability of L Vector
        float LdH = max(L.Dot(H), 0.0f);
        pdf /= (4.0f * (LdH));

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
                     const UnrealMatData& matData,
                     const HitKey::Type& matId)
    {
        // It is impossible to evaluate if object is highly specular
        using namespace TracerConstants;
        bool isSpecular = (Specularity(surface, matData, matId) >= SPECULAR_THRESHOLD);
        if(isSpecular)
        {
            return Zero3f;
        }


        float roughness = (*matData.dRoughness[matId])(surface.uv);
        Vector3 N = ZAxis;
        Vector3 L = GPUSurface::ToTangent(wo, surface.worldToTangent);
        Vector3 V = GPUSurface::ToTangent(wi, surface.worldToTangent);
        Vector3 H = (L + V).Normalize();
        float NdH = abs(N.Dot(H));
        float D = TracerFunctions::DGGX(NdH, roughness);

        return Vector3f(D);

        //// Acquire Parameters
        //// Check if normal mapping is present
        //Vector3 N = ZAxis;
        //if(matData.dNormal[matId])
        //    N = (*matData.dNormal[matId])(surface.uv).Normalize();
        //float roughness = (*matData.dRoughness[matId])(surface.uv);
        //float metallic = (*matData.dMetallic[matId])(surface.uv);
        //float specular = (*matData.dSpecular[matId])(surface.uv);
        //Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);

        //Vector3 L = GPUSurface::ToTangent(wo, surface.worldToTangent);
        //Vector3 V = GPUSurface::ToTangent(wi, surface.worldToTangent);
        //Vector3 H = (L + V).Normalize();
        //// BRDF Calculation
        ////float NdL = max(N.Dot(L), 0.0f);
        ////float NdV = max(N.Dot(V), 0.0f);
        ////float NdH = max(N.Dot(H), 0.0f);
        ////float VdH = max(V.Dot(H), 0.0f);
        ////float LdH = max(L.Dot(H), 0.0f);
        //float NdL = abs(N.Dot(L));
        //float NdV = abs(N.Dot(V));
        //float NdH = abs(N.Dot(H));
        //float VdH = abs(V.Dot(H));
        //float LdH = abs(L.Dot(H));

        //// Edge cases
        //if(NdV == 0.0f || LdH == 0.0f)
        //{
        //    return Zero3;
        //}

        //// Shadowing Term (Schlick Model)
        ////float G = TracerFunctions::GSchlick(NdL, roughness) *
        ////          TracerFunctions::GSchlick(NdV, roughness);
        //float G = 1.0f;
        //// GGX
        //float D = TracerFunctions::DGGX(NdH, roughness);
        //// Fresnel Term (Schlick's Approx)
        //Vector3f f0 = CalculateF0(albedo, metallic, specular);
        ////Vector3f F = TracerFunctions::FSchlick(VdH, f0);
        //Vector3f F = Vector3f(1.0f);
        //// Calculate Radiance
        //// Blend between albedo-black for metallic material
        //Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
        //Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;
        //// Notice that NdL terms are canceled out
        //Vector3f specularTerm = Vector3f(D) * F * G * 0.25f / NdV;
        //specularTerm = (NdV == 0.0f) ? Zero3 : specularTerm;

        //// Blend diffuse term due to metallic
        //return /*diffuseTerm +*/ specularTerm;
    }

    // Does not have emission
    static constexpr auto& IsEmissive   = IsEmissiveFalse<UnrealMatData>;
    static constexpr auto& Emit         = EmitEmpty<UnrealMatData, UVSurface>;
};
