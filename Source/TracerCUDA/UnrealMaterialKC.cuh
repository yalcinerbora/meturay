#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"

#include "Random.cuh"
#include "ImageFunctions.cuh"
#include "MaterialFunctions.h"
#include "TracerFunctions.cuh"
#include "GPUSurface.h"
#include "MetaMaterialFunctions.cuh"

__device__ __forceinline__
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
    __device__ __forceinline__ static
    float Specularity(const UVSurface& surface,
                      const UnrealMatData& matData,
                      const HitKey::Type& matId)
    {
        float roughness = (*matData.dRoughness[matId])(surface.uv);
        float alpha = roughness * roughness;
        return 1.0f - alpha;
    }

    __device__ __forceinline__ static
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
                   RandomGPU& rng,
                   // Constants
                   const UnrealMatData& matData,
                   const HitKey::Type& matId,
                   uint32_t sampleIndex)
    {
        // No medium change
        outMedium = &m;

        // Acquire Parameters
        // Check if normal mapping is present
        Vector3 N = ZAxis;
        if(matData.dNormal[matId])
            N = (*matData.dNormal[matId])(surface.uv);
        float roughness = (*matData.dRoughness[matId])(surface.uv);
        float metallic = (*matData.dMetallic[matId])(surface.uv);
        float specular = (*matData.dSpecular[matId])(surface.uv);
        Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
        // Since we using capital terms alias wi as V
        Vector3 V = GPUSurface::ToTangent(wi, surface.worldToTangent);
        // Sample a H (Half Vector)
        Vector3 H;
        float D = TracerFunctions::DGGXSample(H, pdf, roughness, rng);
        // Gen L (aka. wo)
        Vector3 L = 2.0f * V.Dot(H) * H - V;

        // BRDF Calculation
        float NdL = max(N.Dot(L), 0.0f);
        float NdV = max(N.Dot(V), 0.0f);
        float NdH = max(N.Dot(H), 0.0f);
        float VdH = max(V.Dot(H), 0.0f);
        float LdH = max(L.Dot(H), 0.0f);

        // Edge cases
        if(NdV == 0.0f || LdH == 0.0f)
        {
            pdf = 0.0f;
            return Zero3;
        }

        // GGXSample returns sampling of H Vector
        // convert it to sampling probability of L Vector
        pdf /= (4.0f * (LdH));

        // Shadowing Term (Schlick Model)
        float G = TracerFunctions::GSchlick(NdL, roughness) *
                  TracerFunctions::GSchlick(NdV, roughness);
        // Fresnel Term (Schlick's Approx)
        Vector3f f0 = CalculateF0(albedo, metallic, specular);
        Vector3f F = TracerFunctions::FSchlick(VdH, f0);
        // We need to slightly nudge the ray start
        // to prevent self intersection
        // Normal is on tangent space so convert it to world space
        // Convert Normal to World Space
        Vector3f normalWorld = GPUSurface::ToWorld(N, surface.worldToTangent);
        // Same is true for wo(aka L)
        Vector3f woDir = GPUSurface::ToWorld(L, surface.worldToTangent);
        Vector3f woPos = pos + normalWorld * MathConstants::Epsilon;

        // Calculate Radiance
        // Blend between albedo-black for metallic material
        Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
        Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;
        // Notice that NdL terms are canceled out
        Vector3f specularTerm = D * F * G * 0.25f / NdV;
        specularTerm = (NdV == 0.0f) ? Zero3 : specularTerm;

        // Ray Out
        wo = RayF(woDir, pos);

        // DEBUG
        //if(specularTerm.HasNaN() || woDir.HasNaN() ||
        //   L.HasNaN() || H.HasNaN() ||
        //   wo.getDirection().HasNaN())
        //{
        //    printf("pdf %f, G %f, D %f \n"
        //           "NdL %f, NdV %f, NdH %f, VdH %f, LdH %f\n"
        //           "F %f %f %f\n"
        //           "L: (%f, %f, %f)\n"
        //           "---\n",
        //           pdf, G, D,
        //           NdL, NdV, NdH, VdH, LdH,
        //           F[0], F[1], F[2],
        //           L[0], L[1], L[2]);
        //}

        // PDF is already written
        // Finally return Radiance
        // All Done!
        return diffuseTerm + specularTerm;
    }

    __device__ __forceinline__ static
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
        float roughness = (*matData.dRoughness[matId])(surface.uv);
        Vector3 N = ZAxis;
        if(matData.dNormal[matId])
            N = (*matData.dNormal[matId])(surface.uv);

        Vector3 L = GPUSurface::ToTangent(wo, surface.worldToTangent);
        Vector3 V = GPUSurface::ToTangent(wi, surface.worldToTangent);
        Vector3 H = (L + V).Normalize();

        float NdH = max(N.Dot(H), 0.0f);
        //float sinTheta = sqrt(max(0.0f, 1.0f - NdH * NdH));
        float pdf = TracerFunctions::DGGX(NdH, roughness) * NdH;

        // GGXSample returns sampling of H Vector
        // convert it to sampling probability of L Vector
        float LdH = max(L.Dot(H), 0.0f);
        pdf /= (4.0f * (LdH));

        return pdf;
    }

    __device__ __forceinline__ static
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
        Vector3 N = ZAxis;
        if(matData.dNormal[matId])
            N = (*matData.dNormal[matId])(surface.uv);
        float roughness = (*matData.dRoughness[matId])(surface.uv);
        float metallic = (*matData.dMetallic[matId])(surface.uv);
        float specular = (*matData.dSpecular[matId])(surface.uv);
        Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);

        Vector3 L = GPUSurface::ToTangent(wo, surface.worldToTangent);
        Vector3 V = GPUSurface::ToTangent(wi, surface.worldToTangent);
        Vector3 H = (L + V).Normalize();
        // BRDF Calculation
        float NdL = max(N.Dot(L), 0.0f);
        float NdV = max(N.Dot(V), 0.0f);
        float NdH = max(N.Dot(H), 0.0f);
        float VdH = max(V.Dot(H), 0.0f);
        float LdH = max(L.Dot(H), 0.0f);

        // Edge cases
        if(NdV == 0.0f || LdH == 0.0f)
        {
            return Zero3;
        }

        // Shadowing Term (Schlick Model)
        float G = TracerFunctions::GSchlick(NdL, roughness) *
                  TracerFunctions::GSchlick(NdV, roughness);
        // GGX
        float D = TracerFunctions::DGGX(NdH, roughness);
        // Fresnel Term (Schlick's Approx)
        Vector3f f0 = CalculateF0(albedo, metallic, specular);
        Vector3f F = TracerFunctions::FSchlick(VdH, f0);
        // Calculate Radiance
        // Blend between albedo-black for metallic material
        Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
        Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;
        // Notice that NdL terms are canceled out
        Vector3f specularTerm = D * F * G * 0.25f / NdV;
        specularTerm = (NdV == 0.0f) ? Zero3 : specularTerm;

        // DEBUG
        //if(specularTerm.HasNaN())
        //    printf("G %f, D %f \n"
        //           "NdL %f, NdV %f, NdH %f, VdH %f, LdH %f\n"
        //           "F %f %f %f\n---\n",
        //           G, D,
        //           NdL, NdV, NdH, VdH, LdH,
        //           F[0], F[1], F[2]);

        // Blend diffuse term due to metallic
        return diffuseTerm + specularTerm;
    }

    // Does not have emission
    static constexpr auto& IsEmissive   = IsEmissiveFalse<UnrealMatData>;
    static constexpr auto& Emit         = EmitEmpty<UnrealMatData, UVSurface>;
};

