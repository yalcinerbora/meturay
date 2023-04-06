#pragma once

#include "RayLib/CoordinateConversion.h"

namespace TracerFunctions
{
    __device__ inline
    float FresnelDielectric(float cosIn, float iorIn, float iorOut)
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

    template <class T>
    __device__ inline
    T FresnelConductor(float cosIn, const T& iorIn, const T& kIn)
    {
        // https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#FrConductor
        //
        // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
        // Find sin from trigonometry
        float cosInSqr = cosIn * cosIn;
        float sinInSqr = max(0.0f, 1.0f - cosInSqr);
        float sinInSqr2 = sinInSqr * sinInSqr;

        float sinIn = sqrt(max(sinInSqr, 0.0f));

        T etaSqr = iorIn * iorIn;
        T kSqr = kIn * kIn;

        T diffTerm = etaSqr - kSqr - T(sinInSqr);
        T a2b2 = diffTerm * diffTerm;
        a2b2 += 4.0f * nSqr * kSqr;
        a2b2 = sqrt(max(a2b2, 0.0f)); // Complex Skip
        //
        T a = 0.5f * (a2b2 + diffTerm);
        a = sqrt(max(a, 0.0f));

        T perpendicular = a2b2 - (2.0f * a * cosIn) + cosInSqr;
        perpendicular /= (a2b2 + (2.0f * a * cosIn) + cosInSqr);

        T parallel = cosInSqr * a2b2 - (2.0f * a * cosIn * sinInSqr) + sinInSqr4;
        parallel /= (cosInSqr * a2b2 + (2.0f * a * cosIn * sinInSqr) + sinInSqr4);
        parallel *= perpendicular;
        return (parallel + perpendicular) * 0.5f;
    }

    __device__ inline
    float DGGX(float NdH, float alpha)
    {
        float alphaSqr = alpha * alpha;
        float denom = NdH * NdH * (alphaSqr - 1.0f) + 1.0f;
        denom = denom * denom;
        denom *= MathConstants::Pi;
        float result = (alphaSqr / denom);
        return result;
    }

    __device__ inline
    float LambdaSmith(const Vector3f& vec, float alpha)
    {
        Vector3f vSqr = vec * vec;
        float alphaSqr = alpha * alpha;
        float inner = alphaSqr * (vSqr[0] + vSqr[1]) / vSqr[2];
        float lambda = sqrt(1.0f + inner) - 1.0f;
        lambda *= 0.5f;
        return lambda;
    }

    __device__ inline
    float GSmithSingle(const Vector3f& vec, float alpha)
    {
        return 1.0f / (1.0f + LambdaSmith(vec, alpha));
    }

    __device__ inline
    float GSmithCorralated(const Vector3f& wO,
                           const Vector3f& wI,
                           float alpha)
    {
        // Height correlated mask/shadowing
        return 1.0f / (LambdaSmith(wO, alpha) + LambdaSmith(wI, alpha) + 1.0f);
    }

    __device__ inline
    float GSmithSeperable(const Vector3f& wO,
                          const Vector3f& wI,
                          float alpha)
    {
        return GSmithSingle(wO, alpha) * GSmithSingle(wI, alpha);
    }

    __device__ inline
    float GSchlick(float dot, float alpha)
    {
        //// "Hotness" removal
        //roughness = (roughness + 1) * 0.5f;
        if(dot == 0.0f) return 0;
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

    __device__ inline
    float GeomGGX(float dot, float alpha)
    {
        // Straight from paper
        // https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
        if(dot == 0.0f) return 0;

        float alphaSqr = alpha * alpha;

        float dotSqr = dot;
        float denom = (1.0f - dotSqr) / dotSqr;
        denom *= alphaSqr;
        denom += 1.0f;
        denom = sqrt(denom) + 1;

        return 2.0f / denom;
    }

    __device__ inline
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

    __device__ inline
    float VNDFGGXSmithPDF(const Vector3f V, const Vector3f H, float alpha)
    {
        float VdH = max(0.0f, H.Dot(V));
        float NdH = max(0.0f, H[2]);
        float NdV = max(0.0f, V[2]);
        float D = DGGX(NdH, alpha);
        float GSingle = GSmithSingle(V, alpha);
        float pdf = (NdV == 0.0f) ? 0.0f : (VdH * D * GSingle / NdV);
        return pdf;
    }

    __device__ inline
    Vector3f VNDFGGXSmithSample(float& pdf,
                                const Vector3f& V,
                                float alpha,
                                RNGeneratorGPUI& rng)
    {
        // VNDF Routine straight from the paper
        // https://jcgt.org/published/0007/04/01/
        // G1 is Smith here be careful,
        // Everything is tangent space,
        // So no surface normal is feed to the system,
        // some Dot products (with normal) are thusly represented as
        // X[2] where x is the vector is being dot product with the normal
        //
        // Unlike most of the routines this sampling function
        // consists of multiple functions (namely NDF and Shadowing)
        // because of that, it does not return the value of the function
        // it returns the generated micro-facet normal
        //
        // And finally this routine represents isotropic material
        // a_y ==  a_x == a
        Vector2f xi = rng.Uniform2D();
        // Rename alpha for easier reading
        float a = alpha;
        // Section 3.2 Ellipsoid to Spherical
        Vector3f VHemi = Vector3f(a * V[0], a * V[1], V[2]).Normalize();
        // Section 4.1 Find orthonormal basis in the sphere
        float lensq = Vector2f(VHemi).LengthSqr();
        Vector3f T1 = (lensq > 0) ? Vector3f(-VHemi[1], VHemi[0], 0.0f) / sqrt(lensq) : Vector3f(1, 0, 0);
        Vector3f T2 = Cross(VHemi, T1);
        // Section 4.2 Sampling using projected area
        float r = sqrt(xi[0]);
        float phi = 2.0f * MathConstants::Pi * xi[1];
        float t1 = r * cos(phi);
        float t2 = r * sin(phi);
        float s = 0.5f * (1.0f + VHemi[2]);
        t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
        // Section 4.3: Projection onto hemisphere
        float val = 1.0f - t1 * t1 - t2 * t2;
        Vector3f NHemi = t1 * T1 + t2 * T2 + sqrt(max(0.0f, val)) * VHemi;
        // Section 3.4: Finally back to Ellipsoid
        Vector3f NMicrofacet = Vector3f(a * NHemi[0], a * NHemi[1], max(0.0f, NHemi[2]));
        NMicrofacet.NormalizeSelf();

        // To make it consistent between other functions,
        // we will return PDF of the value that is being returned (micro-facet normal)
        // instead of the L. Convert it to reflected light after this function
        //float VdH = max(0.0f, NMicrofacet.Dot(V));
        //float NdH = max(0.0f, NMicrofacet[2]);
        //float NdV = max(0.0f, V[2]);
        //float D = DGGX(NdH, alpha);
        //float GSingle = GSmithSingle(V, alpha);
        //pdf = (NdV == 0.0f) ? 0.0f : (VdH * D * GSingle / NdV);
        pdf = VNDFGGXSmithPDF(V, NMicrofacet, alpha);
        return NMicrofacet;
    }

    __device__ inline
    float DGGXSample(Vector3& H,
                     float& pdf,
                     float alpha,
                     RNGeneratorGPUI& rng)
    {
        // https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
        // Page 4 it does not include pdf it is included from
        // https://agraphicsguynotes.com/posts/sample_microfacet_brdf/
        // https://www.tobias-franke.eu/log/2014/03/30/notes_on_importance_sampling.html
        float xi0 = rng.Uniform();
        float xi1 = rng.Uniform();

        float a = alpha;
        float aSqr = a * a;

        float phi = 2.0f * MathConstants::Pi * xi0;
        float cosTheta = sqrt(max(0.0f, (1.0f - xi1) / ((aSqr - 1.0f) * xi1 + 1.0f)));
        float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

        // Spherical Coord conversion functions has azimuth as phi so
        // this may look wrong but it is not
        H = Utility::SphericalToCartesianUnit(Vector2f(sin(phi), cos(phi)),
                                              Vector2f(sinTheta, cosTheta));

        // Pdf
        float ggxResult = DGGX(cosTheta, a);
        pdf = cosTheta * ggxResult;
        return ggxResult;
    }

    __device__ inline
    float PowerHeuristic(float n0, float pdf0, float n1, float pdf1)
    {
        // This is power-2 heuristic
        float w0 = n0 * pdf0;
        float w1 = n1 * pdf1;

        return (w0 * w0) / (w0 * w0 + w1 * w1);
    }

    __device__ inline
    float BalanceHeuristic(float n0, float pdf0, float n1, float pdf1)
    {
        float w0 = n0 * pdf0;
        float w1 = n1 * pdf1;
        return w0 / (w0 + w1);
    }

    // Basic Russian Roulette
    __device__ inline
    bool RussianRoulette(Vector3& irradianceFactor,
                         float probFactor, RNGeneratorGPUI& rng)
    {
        // Basic Russian Roulette
        probFactor = HybridFuncs::Clamp(probFactor, 0.005f, 1.0f);
        if(rng.Uniform() >= probFactor)
            return true;
        else irradianceFactor *= (1.0f / probFactor);
        return false;
    }
}