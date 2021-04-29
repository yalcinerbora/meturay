#pragma once

#include "RayLib/CoordinateConversion.h"

namespace TracerFunctions
{
    __device__
    inline float FrenelDielectric(float cosIn, float iorIn, float iorOut)
    {
        // Calculate Sin from Snell's Law
        float sinIn = sqrt(max(0.0f, 1.0f - cosIn * cosIn));
        float sinOut = iorIn / iorOut * sinIn;

        // Total internal reflection
        if(sinOut >= 1.0f) return 1.0f;

        // Frenel Equation
        float cosOut = sqrt(max(0.0f, 1.0f - sinOut * sinOut));

        float parallel = ((iorOut * cosIn - iorIn * cosOut) /
                          (iorOut * cosIn + iorIn * cosOut));
        parallel = parallel * parallel;

        float perpendicular = ((iorIn * cosIn - iorOut * cosOut) /
                               (iorIn * cosIn + iorOut * cosOut));
        perpendicular = perpendicular * perpendicular;

        return (parallel + perpendicular) * 0.5f;
    }

    __device__ 
    inline float GGX(float NdH, float roughness)
    {
        float alpha = roughness * roughness;
        float alphaSqr = alpha * alpha;
        float dDenom = NdH * NdH * (alphaSqr - 1.0f) + 1.0f;
        dDenom = dDenom * dDenom;
        dDenom *= MathConstants::Pi;
        return (alphaSqr / dDenom);
    }

    __device__
    inline float GGXSample(Vector3& H,
                            float& pdf,
                            float roughness,
                            RandomGPU& rng)
    {
        // https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
        // Page 4 it does not include pdf it is included from
        // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/
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
        //float ggxResult = GGX(cosTheta, roughness);
        //pdf = ggxResult * cosTheta;// *sinTheta
        //return ggxResult;

        // Pre-cancel ggx (to avoid NaN's)
        pdf = cosTheta;
        return 1.0f;
    }

    __device__ 
    inline float GSchlick(float dot, float roughness)
    {
        float k = (roughness + 1);
        k = k * k;
        // This is much more verbose than 0.125f
        // and it shoud have same perf
        static constexpr float denom = 1.0f / 8.0f;
        k *= denom;

        return dot / (dot * (1 - k) + k);
    }

    __device__ 
    inline Vector3f FSchlick(float VdH, const Vector3f& f0)
    {
        static constexpr float t0 = -5.55473f;
        static constexpr float t1 = -6.98316f;

        float pw = pow(2.0f, (t0 * VdH - t1) * VdH);
        Vector3f result = (Vector3f(1.0f) - f0) * pw;
        result += f0;
        return result;
    }

    __device__
    inline float PowerHeuristic(int n0, float pdf0, int n1, float pdf1)
    {
        // This is power-2 heuristic
        float w0 = static_cast<float>(n0) * pdf0;
        float w1 = static_cast<float>(n1) * pdf1;

        return (w0 * w0) / (w0 * w0 + w1 * w1);
    }

    // Basic Russian Roulette
    __device__
    inline bool RussianRoulette(Vector3& irradianceFactor,
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