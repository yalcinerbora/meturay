#pragma once

#include "RayLib/Quaternion.h"
#include "RayLib/HybridFunctions.h"

//
// https://therealmjp.github.io/posts/sg-series-part-2-spherical-gaussians-101/
// Spherical Gaussian implementation
class GaussianLobe
{
    public:
    //private:
    Vector3f                            direction;
    float                               amplitude;
    float                               sharpness;

    //
    __host__ __device__ float          Integrate() const;

    protected:
    public:
    // Constructors & Destructor
    __host__ __device__                 GaussianLobe(Vector3f direction,
                                                     float amplitude,
                                                     float sharpness);

    // Combining two lobes
    __host__ __device__ float           Dot(const GaussianLobe&) const;
    __host__ __device__ GaussianLobe    Mult(const GaussianLobe&) const;
    __host__ __device__ GaussianLobe    Convolve(const GaussianLobe&) const;

    // Classic sampling schemes
    template<class RNG>
    __host__ __device__ void            Sample(Vector3f& dir, float& pdf,
                                               RNG& rng) const;
    __host__ __device__ float           Eval(const Vector3f& dir) const;
    __host__ __device__ float           PDF(const Vector3f& dir) const;

    // Some Helpers
    __host__ __device__
    [[nodiscard]] GaussianLobe          Rotate(const QuatF&) const;
    __host__ __device__ GaussianLobe&   RotateSelf(const QuatF&);
    __host__ __device__
    [[nodiscard]] GaussianLobe          Normalize() const;
    __host__ __device__ GaussianLobe&   NormalizeSelf();

    __host__ __device__
    static GaussianLobe                 Interpolate(const GaussianLobe& a,
                                                    const GaussianLobe& b,
                                                    float t);
};

__host__ __device__
inline float GaussianLobe::Integrate() const
{
    float expTerm = 1.0f - exp(-2.0f * sharpness);
    return 2.0f * MathConstants::Pi * (amplitude / sharpness) * expTerm;
}

__host__ __device__
inline GaussianLobe::GaussianLobe(Vector3f direction,
                                  float amplitude,
                                  float sharpness)
    : direction(direction)
    , amplitude(amplitude)
    , sharpness(sharpness)
{}

__host__ __device__
inline GaussianLobe GaussianLobe::Mult(const GaussianLobe& right) const
{
    float lm = sharpness + right.sharpness;
    Vector3f um = (sharpness * direction + right.sharpness * right.direction) / lm;
    float umLength = um.Length();
    float amp = amplitude * right.amplitude * exp(lm * (umLength - 1.0f));

    return GaussianLobe(um / umLength,
                        amp,
                        lm * umLength);
}

__host__ __device__
inline float GaussianLobe::Dot(const GaussianLobe& right) const
{
    float dm = (sharpness * direction +
                right.sharpness * right.direction).Length();

    float expo = expf(dm - sharpness - right.sharpness);
    float other = 1.0f - expf(-2.0f * dm);
    float result = 2.0f * MathConstants::Pi * amplitude * right.amplitude;
    return (result * expo * other) / dm;
}

__host__ __device__
inline GaussianLobe GaussianLobe::Convolve(const GaussianLobe& right) const
{
    // https://dl.acm.org/doi/pdf/10.1145/2366145.2366163
    // Equation 14
    float amp = 4.0f * MathConstants::Pi * amplitude * right.amplitude;
    amp *= exp(-(sharpness + right.sharpness));

    float length = (direction * sharpness + right.direction * sharpness).Length();



    return GaussianLobe(YAxis, 1, 1);
}

__host__ __device__
inline GaussianLobe GaussianLobe::Interpolate(const GaussianLobe& a,
                                              const GaussianLobe& b,
                                              float t)
{
    // Assuming Gaussian lobes are normalized
    // TODO: research this (might not be good)
    Vector3f axis = Cross(a.direction, b.direction).Normalize();
    float angle = acos(a.Dot(b));
    QuatF rot(angle * t, axis);
    Vector3f newDir = rot.ApplyRotation(a.direction);

    float newAmp = HybridFuncs::Lerp(a.amplitude, b.amplitude, t);
    float newSharp = HybridFuncs::Lerp(a.sharpness, b.sharpness, t);
    return GaussianLobe(newDir, newAmp, newSharp);
}

// Classic sampling schemes
template<class RNG>
__host__ __device__
inline void GaussianLobe::Sample(Vector3f& dir,
                                 float& pdf,
                                 RNG& rng) const
{

}

__host__ __device__
inline float GaussianLobe::Eval(const Vector3f& dir) const
{
    return NAN;
}

__host__ __device__
inline float GaussianLobe::PDF(const Vector3f& dir) const
{
    return NAN;
}

[[nodiscard]] __host__ __device__
inline GaussianLobe GaussianLobe::Rotate(const QuatF& q) const
{
    return GaussianLobe(q.ApplyRotation(direction), amplitude, sharpness);
}

__host__ __device__
inline GaussianLobe& GaussianLobe::RotateSelf(const QuatF& q)
{
    direction = q.ApplyRotation(direction);
    return *this;
}

 __host__ __device__
inline GaussianLobe GaussianLobe::Normalize() const
{
    float integral = Integrate();
    GaussianLobe result = *this;
    result.amplitude /= integral;
    return result;
}

__host__ __device__
inline GaussianLobe& GaussianLobe::NormalizeSelf()
{
    float integral = Integrate();
    amplitude /= integral;
    return *this;
}