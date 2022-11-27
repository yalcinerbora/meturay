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
        // Transform one level extra to the normal map tangent space if
        // normal map is available
        QuatF toTangent = surface.worldToTangent;
        if((matData.dNormal[matId] != nullptr))
        {
            Vector3f N = (*matData.dNormal[matId])(surface.uv).Normalize();
            toTangent = toTangent * Quat::RotationBetweenZAxis(N);
        }
        // Since we using capital terms alias wi as V
        Vector3f V = GPUSurface::ToTangent(wi, toTangent);
        // Sample a H (Half Vector)
        Vector3f H = TracerFunctions::VNDFGGXSmithSample(pdf, V, alpha, rng);
        // Reflect the light using microfacet normal
        float VdH = V.Dot(H);
        Vector3f L = 2.0f * VdH * H - V;
        // Pre-check the shadowing term switches
        float LdH = L.Dot(H);
        if(LdH <= 0.0f || // wi is masked
           VdH <= 0.0f || // wo is shadowed
           pdf == 0.0f)
        {
            pdf = 0.0f;
            return Zero3f;
        }

        // VNDFGGXSmithSample returns sampling of H Vector
        // convert it to sampling probability of L Vector
        //printf("pdf
        pdf /= (4.0f * VdH);

        //=======================//
        //   Calculate Specular  //
        //=======================//
        using namespace GPUSurface;
        // Normal Distribution Function (GGX)
        float D = TracerFunctions::DGGX(DotN(H), alpha);
        // Shadowing Term (Smith Model)
        float G = TracerFunctions::GSmithCorralated(V, L, alpha);
        // Fresnel Term (Schlick's Approx)
        Vector3f f0 = CalculateF0(albedo, metallic, specular);
        Vector3f F = TracerFunctions::FSchlick(VdH, f0);
        // Notice that NdL terms are canceled out
        float NdV = DotN(V);
        Vector3f specularTerm = D * F * G * 0.25f / NdV;
        specularTerm = (NdV <= 0.0f) ? Zero3 : specularTerm;

        // Cancel maybe?
        //specularTerm = G * F / TracerFunctions::GSmithSingle(V, alpha);
        //pdf = 1.0f;

        //Vector3f r = specularTerm / pdf;
        //printf("[S]D: %f, G: %f, F: (%f, %f, %f), "
        //       "G[V] %f, VdH %f,"
        //       "totalSpec (%f, %f, %f), pdf %f = (%f, %f, %f)\n",
        //       D, G,
        //       F[0], F[1], F[2],
        //       TracerFunctions::GSmithSingle(V, alpha), VdH,
        //       specularTerm[0], specularTerm[1], specularTerm[2],
        //       pdf, r[0], r[1], r[2]);

        //=======================//
        //   Calculate Diffuse   //
        //=======================//
        // Blend between albedo<->black for metallic material
        float NdL = max(DotN(L), 0.0f);
        Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
        Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;

        //=======================//
        //  Calculate Direction  //
        //=======================//
        // We need to slightly nudge the ray start
        // to prevent self intersection
        // Normal is on tangent space so convert it to world space
        // Convert Tangent Space to World Space
        wo = RayF(GPUSurface::ToSpace(L, toTangent), pos);
        wo.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);
        // PDF is already written
        // Finally return Radiance
        // All Done!
        return /*diffuseTerm +*/ specularTerm;
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
        //// It is impossible to find exact wo <=> wi
        //// with a chance
        //using namespace TracerConstants;
        //bool isSpecular = (Specularity(surface, matData, matId) >= SPECULAR_THRESHOLD);
        //if(isSpecular)
        //{
        //    return 0.0f;
        //}

        float roughness = (*matData.dRoughness[matId])(surface.uv);
        float alpha = roughness * roughness;

        // Convert to Tangent Space
        QuatF toTangentRot = surface.worldToTangent;
        if(matData.dNormal[matId] != nullptr)
        {
            Vector3f N = (*matData.dNormal[matId])(surface.uv).Normalize();
            toTangentRot = Quat::RotationBetweenZAxis(N) * toTangentRot;
        }
        Vector3f L = GPUSurface::ToTangent(wo, toTangentRot);
        Vector3f V = GPUSurface::ToTangent(wi, toTangentRot);
        Vector3f H = (L + V).Normalize();

        float HdV = max(0.0f, H.Dot(V));
        // Use optimized dot product between N here (which just does V[2])
        // but it is more verbose
        using namespace GPUSurface;
        float pdf = (HdV *
                     TracerFunctions::DGGX(DotN(H), alpha) *
                     TracerFunctions::GSmithSingle(V, alpha) / DotN(V));

        return (DotN(V) == 0.0f) ? 0.0f : pdf;
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
        //// It is impossible to evaluate if object is highly specular
        //using namespace TracerConstants;
        //bool isSpecular = (Specularity(surface, matData, matId) >= SPECULAR_THRESHOLD);
        //if(isSpecular)
        //{
        //    return Zero3f;
        //}

        // Acquire Parameters
        // Check if normal mapping is present
        float roughness = (*matData.dRoughness[matId])(surface.uv);
        float metallic = (*matData.dMetallic[matId])(surface.uv);
        float specular = (*matData.dSpecular[matId])(surface.uv);
        Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
        float alpha = roughness * roughness;

        // Convert to Tangent Space
        QuatF toTangentRot = surface.worldToTangent;
        if(matData.dNormal[matId] != nullptr)
        {
            Vector3f N = (*matData.dNormal[matId])(surface.uv).Normalize();
            toTangentRot = Quat::RotationBetweenZAxis(N) * toTangentRot;
        }
        Vector3f L = GPUSurface::ToTangent(wo, toTangentRot);
        Vector3f V = GPUSurface::ToTangent(wi, toTangentRot);
        Vector3f H = (L + V).Normalize();

        //=======================//
        //   Calculate Specular  //
        //=======================//
        using namespace GPUSurface;
        float LdH = L.Dot(H);
        float VdH = V.Dot(H);
        // Normal Distribution Function (GGX)
        float D = TracerFunctions::DGGX(DotN(H), alpha);
        // Shadowing Term (Smith Model)
        float G = TracerFunctions::GSmithCorralated(V, L, alpha);
        G = (LdH <= 0.0f) ? 0.0f : G;
        G = (VdH <= 0.0f) ? 0.0f : G;
        // Fresnel Term (Schlick's Approx)
        Vector3f f0 = CalculateF0(albedo, metallic, specular);
        Vector3f F = TracerFunctions::FSchlick(VdH, f0);
        // Notice that NdL terms are canceled out
        float NdV = DotN(V);
        Vector3f specularTerm = D * F * G * 0.25f / NdV;
        specularTerm = (NdV <= 0.0f) ? Zero3 : specularTerm;

        //printf("[E]D: %f, G: %f, F: (%f, %f, %f), totalSpec (%f, %f, %f)\n",
        //       D, G, F[0], F[1], F[2],
        //       specularTerm[0], specularTerm[1],
        //       specularTerm[2]);

        //=======================//
        //   Calculate Diffuse   //
        //=======================//
        // Blend between albedo<->black for metallic material
        float NdL = max(DotN(L), 0.0f);
        Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
        Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;

        // All Done!
        return /*diffuseTerm +*/ specularTerm;
    }

    // Does not have emission
    static constexpr auto& IsEmissive   = IsEmissiveFalse<UnrealMatData>;
    static constexpr auto& Emit         = EmitEmpty<UnrealMatData, UVSurface>;
};
