#pragma once

#include "MaterialDataStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"

#include "Random.cuh"
#include "ImageFunctions.cuh"
#include "MaterialFunctions.cuh"
#include "TracerFunctions.cuh"
#include "GPUSurface.h"

__device__ __forceinline__
Vector3 CalculateF0(const Vector3f& baseAlbedo, float metallic, float specular)
{
    // https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
    // Utilizing proposed specular value to blend base color with specular parameter

    static constexpr float SpecularMax = 0.08f;
    specular *= SpecularMax;

    return Vector3f::Lerp(Vector3f(specular), baseAlbedo, metallic);
}

__device__ __forceinline__
float UnrealSpecularity(const UVSurface& surface,
                        const UnrealMatData& matData,
                        const HitKey::Type& matId)
{
    float roughness = (*matData.dRoughness[matId])(surface.uv);
    float alpha = roughness * roughness;
    return 1.0f - alpha;
}

__device__ __forceinline__
Vector3 UnrealSample(// Sampled Output
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
   /* albedo = N;*/
    // Since we using capital terms alias wi as V
    Vector3 V = GPUSurface::ToTangent(wi, surface.worldToTangent);
    // Sample a H (Half Vector)
    Vector3 H;
    float D = TracerFunctions::DGGXSample(H, pdf, roughness, rng);
    //Vector2 xi(GPUDistribution::Uniform<float>(rng),
    //           GPUDistribution::Uniform<float>(rng));
    //Vector3 H = HemiDistribution::HemiCosineCDF(xi, pdf);
    //float D = TracerFunctions::GGX(max(N.Dot(H), 0.0f),
    //                               roughness);
    
    // Gen L (aka. wo)
    Vector3 L = 2.0f * V.Dot(H) * H - V;

    // BRDF Calculation
    float NdL = max(N.Dot(L), 0.0f);
    float NdV = max(N.Dot(V), 0.0f);
    float NdH = max(N.Dot(H), 0.0f);
    float VdH = max(V.Dot(H), 0.0f);
    float LdH = max(V.Dot(H), 0.0f);
    
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
    // Frenel Term (Schlick's Approx)
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

    //if(isnan(G)) printf("S: G Term NANn");
    //if(F.HasNaN()) printf("S: F Term NANn");
    //if(isnan(D)) printf("S: D Term NANn");

    // Calculate Radiance    
    // Blend between albedo-black for metallic material
    Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
    Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;
    // Notice that NdL terms are cancelled out
    Vector3f specularTerm = D * F * G * 0.25f / NdV;

    if(specularTerm.HasNaN())
    printf("pdf %f, G %f, D %f \n"
           "NdL %f, NdV %f, NdH %f, VdH %f, LdH %f\n"
           "F %f %f %f\n---\n",
           pdf, G, D, 
           NdL, NdV, NdH, VdH, LdH,
           F[0], F[1], F[2]);

    // Ray Out
    wo = RayF(woDir, pos);
    // PDF is already written

    // Finally Radiance
    // All Done!
    return diffuseTerm + specularTerm;
}

__device__ __forceinline__
float UnrealPdf(// Input
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
    float pdf = TracerFunctions::DGGX(NdH, roughness) * NdH;
    return pdf;
}

__device__ __forceinline__
Vector3 UnrealEvaluate(// Input
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
    float metallic = (*matData.dMetallic[matId])(surface.uv);
    float specular = (*matData.dSpecular[matId])(surface.uv);
    Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);

    Vector3 N = ZAxis;
    if(matData.dNormal[matId])
        N = (*matData.dNormal[matId])(surface.uv);
    Vector3 L = GPUSurface::ToTangent(wo, surface.worldToTangent);
    Vector3 V = GPUSurface::ToTangent(wi, surface.worldToTangent);
    Vector3 H = (L + V).Normalize();
    /*albedo = N;*/
    // BRDF Calculation
    float NdL = max(N.Dot(L), 0.0f);
    float NdV = max(N.Dot(V), 0.0f);
    float NdH = max(N.Dot(H), 0.0f);
    float VdH = max(V.Dot(H), 0.0f);
    float LdH = max(V.Dot(H), 0.0f);

    // Edge cases
    if(NdV == 0.0f || LdH == 0.0f)
    {    
        return Zero3;
    }

    // GGX
    float D = TracerFunctions::DGGX(NdH, roughness);
    // Shadowing Term (Schlick Model)    
    float G = TracerFunctions::GSchlick(NdL, roughness) * 
              TracerFunctions::GSchlick(NdV, roughness);
    // Frenel Term (Schlick's Approx)
    Vector3f f0 = CalculateF0(albedo, metallic, specular);
    Vector3f F = TracerFunctions::FSchlick(VdH, f0);
    // Calculate Radiance    
    // Blend between albedo-black for metallic material
    Vector3f diffuseAlbedo = (1.0f - metallic) * albedo;
    Vector3f diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi;
    // Notice that NdL terms are cancelled out
    Vector3f specularTerm = D * F * G * 0.25f / NdV;
    specularTerm = (NdV == 0.0f) ? Zero3 : specularTerm;
    // Blend diffuse term due to metallic
    if(specularTerm.HasNaN())
        printf("G %f, D %f \n"
               "NdL %f, NdV %f, NdH %f, VdH %f, LdH %f\n"
               "F %f %f %f\n---\n",
               G, D,
               NdL, NdV, NdH, VdH, LdH,
               F[0], F[1], F[2]);

    return diffuseTerm + specularTerm;
}