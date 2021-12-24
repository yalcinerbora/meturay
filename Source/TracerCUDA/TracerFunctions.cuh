#pragma once

#include "RayLib/CoordinateConversion.h"

namespace TracerFunctions
{
    __device__ __forceinline__
    float FrenelDielectric(float cosIn, float iorIn, float iorOut)
    {
        // Calculate Sin from Snell's Law
        float sinIn = sqrt(max(0.0f, 1.0f - cosIn * cosIn));
        float sinOut = iorIn / iorOut * sinIn;

        // Total internal reflection
        if(sinOut >= 1.0f) return 1.0f;

        // Fresnel Equation
        float cosOut = sqrt(max(0.0f, 1.0f - sinOut * sinOut));

        float parallel = ((iorOut * cosIn - iorIn * cosOut) /
                          (iorOut * cosIn + iorIn * cosOut));
        parallel = parallel * parallel;

        float perpendicular = ((iorIn * cosIn - iorOut * cosOut) /
                               (iorIn * cosIn + iorOut * cosOut));
        perpendicular = perpendicular * perpendicular;

        return (parallel + perpendicular) * 0.5f;
    }

    __device__ __forceinline__
    float FrenelConductor(float cosIn, float iorIn, float kIn)
    {
        // https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#FrConductor
        //
        // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
        // Find sin from trigonometry
        float cosInSqr = cosIn * cosIn;
        float sinInSqr = max(0.0f, 1.0f - cosInSqr);
        float sinInSqr4 = sinInSqr * sinInSqr;
        float sinIn = sqrt(max(sinInSqr, 0.0f));

        float nSqr = iorIn * iorIn;
        float kSqr = kIn * kIn;

        float a2b2 = (nSqr - kSqr - sinInSqr);
        a2b2 *= a2b2;
        a2b2 += 4.0f * nSqr * kSqr;
        a2b2 = sqrt(max(a2b2, 0.0f)); // Complex Skip
        //
        float a = 0.5f * (a2b2 + (nSqr - kSqr - sinInSqr));
        a = sqrt(max(a, 0.0f));

        float perpendicular = a2b2 - (2.0f * a * cosIn) + cosInSqr;
        perpendicular /= (a2b2 + (2.0f * a * cosIn) + cosInSqr);

        float parallel = cosInSqr * a2b2 - (2.0f * a * cosIn * sinInSqr) + sinInSqr4;
        parallel /= (cosInSqr * a2b2 + (2.0f * a * cosIn * sinInSqr) + sinInSqr4);
        parallel *= perpendicular;


        return (parallel + perpendicular) * 0.5f;
    }

    __device__ __forceinline__
    float DGGX(float NdH, float roughness)
    {
        float alpha = roughness * roughness;
        float alphaSqr = alpha * alpha;
        float denom = NdH * NdH * (alphaSqr - 1.0f) + 1.0f;
        denom = denom * denom;
        denom *= MathConstants::Pi;

        if(abs(denom) < MathConstants::SmallEpsilon)
            return 0.0f;
        float result = (alphaSqr / denom);
        return isnan(result) ? 0.0f : result;
    }

    __device__ __forceinline__
    float DGGXSample(Vector3& H,
                     float& pdf,
                     float roughness,
                     RandomGPU& rng)
    {
        // https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
        // Page 4 it does not include pdf it is included from
        // https://agraphicsguynotes.com/posts/sample_microfacet_brdf/
        // https://www.tobias-franke.eu/log/2014/03/30/notes_on_importance_sampling.html

        float xi0 = GPUDistribution::Uniform<float>(rng);
        float xi1 = GPUDistribution::Uniform<float>(rng);

        float a = roughness * roughness;
        float aSqr = a * a;

        float phi = 2.0f * MathConstants::Pi * xi0;
        float cosTheta = sqrt(max(0.0f, (1.0f - xi1) / ((aSqr - 1.0f) * xi1 + 1.0f)));
        float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

        // Spherical Coord conversion functions has azimuth as phi so
        // this may look wrong but it is not
        H = Utility::SphericalToCartesianUnit(Vector2f(sin(phi), cos(phi)),
                                              Vector2f(sinTheta, cosTheta));

        // Pdf
        float ggxResult = DGGX(cosTheta, roughness);
        pdf = ggxResult * cosTheta;
        return ggxResult;
    }

    __device__ __forceinline__
    float GSchlick(float dot, float roughness)
    {
        //// "Hotness" removal
        //roughness = (roughness + 1) * 0.5f;

        if(dot == 0.0f) return 0;
        float alpha = roughness * roughness;

        // Unreal Version
        // This is much more verbose than 0.125f
        // and it should have same perf
        //static constexpr float denom = 1.0f / 8.0f;
        float k = alpha * 0.5f;
        return dot / (dot * (1 - k) + k);

        //// Straight from the paper
        //// Schlick 1994 [An Inexpensive BRDF Model for Physically - based Rendering]
        //constexpr float PiOvrTwo = MathConstants::Sqrt2 / MathConstants::SqrtPi;
        //float k = alpha * PiOvrTwo;
        //return dot / (dot * (1 - k) + k);
    }

    __device__ __forceinline__
    float GeomGGX(float dot, float roughness)
    {
        // Straight from paper
        // https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
        if(dot == 0.0f) return 0;

        float alpha = roughness * roughness;
        float alphaSqr = alpha * alpha;

        float dotSqr = dot;
        float denom = (1.0f - dotSqr) / dotSqr;
        denom *= alphaSqr;
        denom += 1.0f;
        denom = sqrt(denom) + 1;

        return 2.0f / denom;
    }

    __device__ __forceinline__
    Vector3f FSchlick(float VdH, const Vector3f& f0)
    {
        // Unreal Version from their course notes
        //static constexpr float t0 = -5.55473f;
        //static constexpr float t1 = -6.98316f;
        //float pw = pow(2.0f, (t0 * VdH - t1) * VdH);
        //Vector3f result = (Vector3f(1.0f) - f0) * pw;
        //result += f0;
        //return result;

        // Classic Schlick' Approx
        float pwTerm = 1.0f - VdH;
        float pw5 = pwTerm * pwTerm;
        pw5 *= pw5;
        pw5 *= pwTerm;

        Vector3f result = (Vector3f(1.0f) - f0) * pw5;
        result += f0;
        return result;

    }

    __device__ __forceinline__
    float PowerHeuristic(int n0, float pdf0, int n1, float pdf1)
    {
        // This is power-2 heuristic
        float w0 = static_cast<float>(n0) * pdf0;
        float w1 = static_cast<float>(n1) * pdf1;

        return (w0 * w0) / (w0 * w0 + w1 * w1);
    }

    // Basic Russian Roulette
    __device__ __forceinline__
    bool RussianRoulette(Vector3& irradianceFactor,
                         float probFactor, RandomGPU& rng)
    {
        // Basic Russian Roulette
        probFactor = HybridFuncs::Clamp(probFactor, 0.005f, 1.0f);
        if(GPUDistribution::Uniform<float>(rng) >= probFactor)
            return true;
        else irradianceFactor *= (1.0f / probFactor);
        return false;
    }
}