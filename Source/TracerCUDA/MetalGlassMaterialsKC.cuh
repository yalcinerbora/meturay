#include "MaterialDataStructs.h"

#include "RayLib/HemiDistribution.h"

#include "RNGenerator.h"
#include "MaterialFunctions.h"
#include "GPUSurface.h"
#include "MetaMaterialFunctions.cuh"
#include "TracerFunctions.cuh"

struct MetalMatFuncs
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
                   const UVSurface& surface,
                   // I-O
                   RNGeneratorGPUI& rng,
                   // Constants
                   const MetalMatData& matData,
                   const HitKey::Type& matId,
                   uint32_t sampleIndex)
    {
        // No medium change
        outMedium = &m;

        // Ray Selection
        const Vector3& position = pos;
        float roughness = matData.dRoughness[matId];
        float alpha = roughness * roughness;
        Vector3f eta = matData.dEta[matId];
        Vector3f k = matData.dK[matId];
        Vector3f specular = matData.dSpecular[matId];

        Vector3f V = GPUSurface::ToTangent(wi, surface.worldToTangent);

        // Sample a H (Half Vector)
        Vector3f H = TracerFunctions::VNDFGGXSmithSample(pdf, V, alpha, rng);
        // Reflect the light using microfacet normal
        float VdH = max(0.0f, V.Dot(H));
        Vector3f L = 2.0f * VdH * H - V;

        // VNDFGGXSmithSample returns sampling of H Vector
        // convert it to sampling probability of L Vector
        pdf = (VdH == 0.0f) ? 0.0f : pdf / (4.0f * VdH);

        // Calculations
        using namespace GPUSurface;
        float LdH = max(0.0f, L.Dot(H));
        float NdH = max(0.0f, DotN(H));
        float NdV = max(0.0f, DotN(V));

        Vector3f F = TracerFunctions::FresnelConductor(VdH, eta, k);
        F *= specular;

        // Normal Distribution Function (GGX)
        float D = TracerFunctions::DGGX(NdH, alpha);
        // Shadowing Term (Smith Model)
        float G = TracerFunctions::GSmithCorralated(V, L, alpha);
        G = (LdH == 0.0f) ? 0.0f : G;
        G = (VdH == 0.0f) ? 0.0f : G;

        // Notice that NdL terms are canceled out
        Vector3f specularTerm = D * F * G * 0.25f / NdV;
        specularTerm = (NdV == 0.0f) ? Zero3 : specularTerm;

        // Edge case D is unstable since alpha is too small
        // fall back to cancelled version
        if(isinf(D) ||  // alpha is small
           isnan(D))    // alpha is zero
        {
            specularTerm = G * F;
            if(TracerFunctions::GSmithSingle(V, alpha) != 0.0f)
                specularTerm /= TracerFunctions::GSmithSingle(V, alpha);
            else
                specularTerm = Vector3f(0.0f);

            if(specularTerm.HasNaN() ||
               specularTerm > Vector3f(1000.0f))
                printf("(%f, %f, %f), G:%f, F:(%f, %f, %f), "
                       "GS: %f\n",
                       specularTerm[0], specularTerm[1], specularTerm[2],
                       G, F[0], F[1], F[2], TracerFunctions::GSmithSingle(V, alpha));

            pdf = 1.0f;
        }

        if(specularTerm.HasNaN() ||
           specularTerm > Vector3f(1000.0f) ||
           isnan(pdf) || isinf(pdf))
            printf("++(%f, %f, %f), D: %f, G:%f, F:(%f, %f, %f), "
                   "GS: %f, pdf %f\n",
                   specularTerm[0], specularTerm[1], specularTerm[2],
                   D, G, F[0], F[1], F[2], TracerFunctions::GSmithSingle(V, alpha), pdf);

        // Ray out
        wo = RayF(GPUSurface::ToSpace(L, surface.worldToTangent), position);
        wo.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);

        return specularTerm;
    }

    __device__ inline static
    float Pdf(const Vector3& wo,
              const Vector3& wi,
              const Vector3& pos,
              const GPUMediumI& m,
              //
              const UVSurface& surface,
              // Constants
              const MetalMatData& matData,
              const HitKey::Type& matId)
    {
        Vector3f L = GPUSurface::ToTangent(wo, surface.worldToTangent);
        Vector3f V = GPUSurface::ToTangent(wi, surface.worldToTangent);
        Vector3f H = (L + V).Normalize();

        float roughness = matData.dRoughness[matId];
        float alpha = roughness * roughness;

        // Use optimized dot product between N here (which just does V[2])
        // but it is more verbose
        float NdH = max(0.0f, GPUSurface::DotN(H));
        float VdH = max(0.0f, V.Dot(H));
        float pdfSpecular = TracerFunctions::VNDFGGXSmithPDF(V, H, alpha);
        float D = TracerFunctions::DGGX(NdH, alpha);
        pdfSpecular = (isnan(D) || isinf(D)) ? 0.0f : pdfSpecular;
        pdfSpecular = (VdH == 0.0f) ? 0.0f : pdfSpecular / (4.0f * VdH);
        return pdfSpecular;
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
                     const MetalMatData& matData,
                     const HitKey::Type& matId)
    {
        float roughness = matData.dRoughness[matId];
        float alpha = roughness * roughness;

        Vector3f eta = matData.dEta[matId];
        Vector3f k = matData.dK[matId];
        Vector3f specular = matData.dSpecular[matId];

        Vector3f L = GPUSurface::ToTangent(wo, surface.worldToTangent);
        Vector3f V = GPUSurface::ToTangent(wi, surface.worldToTangent);
        Vector3f H = (L + V).Normalize();

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
        // Fresnel Exact
        Vector3f F = TracerFunctions::FresnelConductor(VdH, eta, k);
        F *= specular;
        // Notice that NdL terms are canceled out
        Vector3f specularTerm = D * F * G * 0.25f / NdV;
        specularTerm = (NdV == 0.0f) ? Zero3 : specularTerm;
    }

    // Does not have emission
    static constexpr auto& IsEmissive   = IsEmissiveFalse<MetalMatData>;
    static constexpr auto& Emit         = EmitEmpty<MetalMatData, UVSurface>;
    static constexpr auto& Specularity  = SpecularityDiffuse<MetalMatData, UVSurface>;
};

//class GlassMatFuncs
//{
//    __device__ inline static
//        Vector3 Sample(// Sampled Output
//                       RayF& wo,
//                       float& pdf,
//                       const GPUMediumI*& outMedium,
//                       // Input
//                       const Vector3& wi,
//                       const Vector3& pos,
//                       const GPUMediumI& m,
//                       //
//                       const UVSurface& surface,
//                       // I-O
//                       RNGeneratorGPUI& rng,
//                       // Constants
//                       const LambertMatData& matData,
//                       const HitKey::Type& matId,
//                       uint32_t sampleIndex)
//    {
//        // No medium change
//        outMedium = &m;
//
//        // Ray Selection
//        const Vector3& position = pos;
//        Vector3 normal = ZAxis;
//        // Check if tangent space normal is avail
//        if(matData.dNormal[matId])
//            normal = (*matData.dNormal[matId])(surface.uv).Normalize();
//
//        // Generate New Ray Direction (This is in tangent space)
//        Vector2 xi(rng.Uniform(), rng.Uniform());
//        Vector3 direction = HemiDistribution::HemiCosineCDF(xi, pdf);
//        direction.NormalizeSelf();
//
//        // Cos Theta
//        float nDotL = max(normal.Dot(direction), 0.0f);
//        // Ray out
//        wo = RayF(GPUSurface::ToSpace(direction, surface.worldToTangent), position);
//        wo.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);
//
//        // Radiance Calculation
//        const Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
//        //printf("%f, %f, %f\n", albedo[0], albedo[1], albedo[2]);
//        return nDotL * albedo * MathConstants::InvPi;
//    }
//
//    __device__ inline static
//    float Pdf(const Vector3& wo,
//                  const Vector3& wi,
//                  const Vector3& pos,
//                  const GPUMediumI& m,
//                  //
//                  const UVSurface& surface,
//                  // Constants
//                  const LambertMatData& matData,
//                  const HitKey::Type& matId)
//    {
//        Vector3 normal = ZAxis;
//        // Check if tangent space normal is avail
//        if(matData.dNormal[matId])
//            normal = (*matData.dNormal[matId])(surface.uv).Normalize();
//        normal = GPUSurface::ToSpace(normal, surface.worldToTangent);
//
//        float pdf = max(wo.Dot(normal), 0.0f);
//        pdf *= MathConstants::InvPi;
//        return pdf;
//    }
//
//    __device__ inline static
//    Vector3 Evaluate(// Input
//                         const Vector3& wo,
//                         const Vector3& wi,
//                         const Vector3& pos,
//                         const GPUMediumI& m,
//                         //
//                         const UVSurface& surface,
//                         // Constants
//                         const LambertMatData& matData,
//                         const HitKey::Type& matId)
//    {
//        Vector3 normal = ZAxis;
//        // Check if tangent space normal is avail
//        if(matData.dNormal[matId])
//            normal = (*matData.dNormal[matId])(surface.uv).Normalize();
//
//        // Calculate lightning in world space since
//        // wo is already in world space
//        normal = GPUSurface::ToSpace(normal, surface.worldToTangent);
//
//        float nDotL = max(normal.Dot(wo), 0.0f);
//        const Vector3f albedo = (*matData.dAlbedo[matId])(surface.uv);
//        return nDotL * albedo * MathConstants::InvPi;
//    }
//
//    // Does not have emission
//    static constexpr auto& IsEmissive = IsEmissiveFalse<LambertMatData>;
//    static constexpr auto& Emit = EmitEmpty<LambertMatData, UVSurface>;
//    static constexpr auto& Specularity = SpecularityDiffuse<LambertMatData, UVSurface>;
//};