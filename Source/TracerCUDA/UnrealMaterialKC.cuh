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

    //__device__ inline static
    //bool IsSpecular(const UVSurface& surface,
    //                const UnrealMatData& matData,
    //                const HitKey::Type& matId)
    //{
    //    using namespace TracerConstants;
    //    return (Specularity(surface, matData, matId) >= SPECULAR_THRESHOLD);
    //}

    __device__ inline static
    float MisRatio(float metallic, float roughness)
    {
        // Return Diffuse Ratio
        //return 0.5f;
        // Hand made MIS ratio
        float alpha = roughness * roughness;
        float sqrtAlpha = roughness;
        float alphaSqr = alpha * alpha;
        float ratio = (1.0f - metallic * metallic) * (metallic * alphaSqr +
                                                      (1 - metallic) * sqrtAlpha);

        // Don't force full ratio to one of the sampling method
        static constexpr float minRatio = 0.05f;
        static constexpr float maxRatio = 0.95f;
        return HybridFuncs::Lerp(minRatio, maxRatio, ratio);
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

        // Calculate the ratio between diffuse/specular
        float misRatio = MisRatio(metallic, roughness);
        float xi = rng.Uniform();
        bool isSampleDiffuse = (xi < misRatio);

        // MIS
        float VdH;
        Vector3f H, L;
        float pdfSelected, pdfOther;
        if(isSampleDiffuse)
        {
            // Sample Diffuse
            L = HemiDistribution::HemiCosineCDF(rng.Uniform2D(),
                                                pdfSelected);
            H = (V + L).Normalize();
            VdH = max(0.0f, V.Dot(H));

            pdfOther = TracerFunctions::VNDFGGXSmithPDF(V, H, alpha);
            // VNDFGGXSmithSample returns sampling of H Vector
            // convert it to sampling probability of L Vector
            pdfOther /= (4.0f * VdH);
        }
        else
        {
            // Sample Specular
            // Sample a H (Half Vector)
            H = TracerFunctions::VNDFGGXSmithSample(pdfSelected, V, alpha, rng);
            // Reflect the light using microfacet normal
            VdH = max(0.0f, V.Dot(H));
            L = 2.0f * VdH * H - V;

            pdfOther = max(0.0f, GPUSurface::DotN(L)) * MathConstants::InvPi;
            // VNDFGGXSmithSample returns sampling of H Vector
            // convert it to sampling probability of L Vector
            pdfSelected /= (4.0f * VdH);

            misRatio = 1.0f - misRatio;
        }
        float& pdfSpecular = (isSampleDiffuse) ? pdfOther : pdfSelected;
        pdfSpecular = (VdH == 0.0f) ? 0.0f : pdfSpecular;

        //=======================//
        //   Calculate Specular  //
        //=======================//
        using namespace GPUSurface;
        float LdH = max(0.0f, L.Dot(H));
        float NdH = max(0.0f, DotN(H));
        float NdV = max(0.0f, DotN(V));
        // Normal Distribution Function (GGX)
        float D = TracerFunctions::DGGX(NdH, alpha);
        // Shadowing Term (Smith Model)
        float G = TracerFunctions::GSmithCorralated(V, L, alpha);
        G = (LdH == 0.0f) ? 0.0f : G;
        G = (VdH == 0.0f) ? 0.0f : G;
        // Fresnel Term (Schlick's Approx)
        Vector3f f0 = CalculateF0(albedo, metallic, specular);
        Vector3f F = TracerFunctions::FSchlick(VdH, f0);
        // Notice that NdL terms are canceled out
        Vector3f specularTerm = D * F * G * 0.25f / NdV;
        specularTerm = (NdV == 0.0f) ? Zero3 : specularTerm;

        // Edge case D is unstable since alpha is too small
        // fall back to cancelled version
        if(isinf(D) ||  // alpha is small
           isnan(D))    // alpha is zero
        {
            specularTerm = G * F / TracerFunctions::GSmithSingle(V, alpha);
            // Don't forget that we are using MIS
            // If we sampled using Specular, pdf is one
            // since we are the sampler
            // on the other hand, diffuse can not sample this value
            pdfSpecular = (isSampleDiffuse) ? 0.0f : 1.0f;
        }
        //=======================//
        //   Calculate Diffuse   //
        //=======================//
        // Blend between albedo<->black for metallic material
        float NdL = max(0.0f, DotN(L));
        Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
        Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;

        //==============================//
        // Multiple Importance Sampling //
        //==============================//
        // Doing "one-sample" MIS here
        //float misWeight = TracerFunctions::BalanceHeuristic(1, pdfSelected, 1, pdfOther);
        float misWeight = TracerFunctions::BalanceHeuristic(misRatio, pdfSelected,
                                                            1 - misRatio, pdfOther);
        pdf = (pdfSelected == 0.0f) ? 0.0f : (misRatio * pdfSelected / misWeight);
        if(isnan(pdf))
        {
            Vector3f r = specularTerm / pdf;
            printf("[S]D: %f, G: %f, F: (%f, %f, %f), "
                   "G[V] %f, VdH %f,"
                   "totalSpec (%f, %f, %f), "
                   "[%s] pdf %f, pdfSelected %f, pdfOther %f\n",
                   D, G,
                   F[0], F[1], F[2],
                   TracerFunctions::GSmithSingle(V, alpha), VdH,
                   specularTerm[0], specularTerm[1], specularTerm[2],
                   (isSampleDiffuse) ? "Diffuse "
                                     : "Specular",
                   pdf, pdfSelected, pdfOther);
        }
        if(pdf < 0.0f)
        {
            printf("[%s] pdf %f, pdfSelected %f, pdfOther %f\n",
                   (isSampleDiffuse) ? "Diffuse " : "Specular",
                   pdf, pdfSelected, pdfOther);
        }

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
        return diffuseTerm + specularTerm;
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
        float metallic = (*matData.dMetallic[matId])(surface.uv);
        float roughness = (*matData.dRoughness[matId])(surface.uv);
        float alpha = roughness * roughness;

        // Convert to Tangent Space
        QuatF toTangentRot = surface.worldToTangent;
        if(matData.dNormal[matId] != nullptr)
        {
            Vector3f N = (*matData.dNormal[matId])(surface.uv).Normalize();
            toTangentRot = toTangentRot * Quat::RotationBetweenZAxis(N);
        }
        Vector3f L = GPUSurface::ToTangent(wo, toTangentRot);
        Vector3f V = GPUSurface::ToTangent(wi, toTangentRot);
        Vector3f H = (L + V).Normalize();

        // Use optimized dot product between N here (which just does V[2])
        // but it is more verbose
        float NdH = max(0.0f, GPUSurface::DotN(H));
        float pdfSpecular = TracerFunctions::VNDFGGXSmithPDF(V, H, alpha);
        float D = TracerFunctions::DGGX(NdH, alpha);
        pdfSpecular = (isnan(D) || isinf(D)) ? 0.0f : pdfSpecular;

        float pdfDiffuse = max(0.0f, GPUSurface::DotN(L)) * MathConstants::InvPi;

        float misRatio = MisRatio(metallic, roughness);
        float pdf = (misRatio * pdfDiffuse +
                     (1.0f - misRatio) * pdfSpecular);

        return pdf;
        //return pdfSpecular;
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
            toTangentRot = toTangentRot * Quat::RotationBetweenZAxis(N);
        }
        Vector3f L = GPUSurface::ToTangent(wo, toTangentRot);
        Vector3f V = GPUSurface::ToTangent(wi, toTangentRot);
        Vector3f H = (L + V).Normalize();

        //=======================//
        //   Calculate Specular  //
        //=======================//
        using namespace GPUSurface;
        float LdH = max(0.0f, L.Dot(H));
        float VdH = max(0.0f, V.Dot(H));
        float NdH = max(0.0f, DotN(H));
        float NdV = max(0.0f, DotN(V));
        // Normal Distribution Function (GGX)
        float D = TracerFunctions::DGGX(NdH, alpha);
        // NDF could exceed the precision (alpha is small) or returns nan
        // (alpha is zero).
        // Assume geometry term was going to zero out the contribution
        // and zero here as well.
        D = (isnan(D) || isinf(D)) ? 0.0f : D;
        // Shadowing Term (Smith Model)
        float G = TracerFunctions::GSmithCorralated(V, L, alpha);
        G = (LdH == 0.0f) ? 0.0f : G;
        G = (VdH == 0.0f) ? 0.0f : G;
        // Fresnel Term (Schlick's Approx)
        Vector3f f0 = CalculateF0(albedo, metallic, specular);
        Vector3f F = TracerFunctions::FSchlick(VdH, f0);
        // Notice that NdL terms are canceled out
        Vector3f specularTerm = D * F * G * 0.25f / NdV;
        specularTerm = (NdV == 0.0f) ? Zero3 : specularTerm;

        //printf("[E]D: %f, G: %f, F: (%f, %f, %f), totalSpec (%f, %f, %f)\n",
        //       D, G, F[0], F[1], F[2],
        //       specularTerm[0], specularTerm[1],
        //       specularTerm[2]);

        //=======================//
        //   Calculate Diffuse   //
        //=======================//
        // Blend between albedo<->black for metallic material
        float NdL = max(0.0f, DotN(L));
        Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
        Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;

        // All Done!
        return diffuseTerm + specularTerm;
    }

    // Does not have emission
    static constexpr auto& IsEmissive   = IsEmissiveFalse<UnrealMatData>;
    static constexpr auto& Emit         = EmitEmpty<UnrealMatData, UVSurface>;
};
