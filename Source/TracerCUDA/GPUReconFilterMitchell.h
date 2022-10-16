#pragma once

#include "GPUReconFilter.h"
#include "TypeTraits.h"

#include "RayLib/TracerError.h"

// Mitchell-Netravali reconstruction filter  implementation
// from PBRT book

class GPUMitchellFilterFunctor
{
    private:
        float           negAlpha;
        Vector2f        radius;
        Vector2f        radiusRecip;
        // Polynomial Coefficients
        Vector4f        coeffs01;
        Vector4f        coeffs12;


        __device__ __host__
        float           Mitchell1D(float x) const;

    public:
        // Constructors & Destructor
                        GPUMitchellFilterFunctor() = default;
                        GPUMitchellFilterFunctor(float radius, float b, float c);
                        ~GPUMitchellFilterFunctor() = default;

        __device__ __host__
        float           operator()(const Vector2f& pixCoord,
                                   const Vector2f& sampleCoord) const;
};

class GPUReconFilterMitchell : public GPUReconFilter
{
    public:
        static const char*              TypeName() { return "Mitchell"; }
        using FilterFunctor             = GPUMitchellFilterFunctor;

        static constexpr const char*    B_OPTION_NAME = "b";
        static constexpr const char*    C_OPTION_NAME = "c";
    private:
        FilterFunctor                   filter;

    protected:
    public:
        // Constructors & Destructor
                    GPUReconFilterMitchell(float filterRadius, Options filterOptions);
                    ~GPUReconFilterMitchell() = default;

        //
        const char* Type() const override { return TypeName(); }

        void        FilterToImg(ImageMemory&,
                                const Vector4f* dValues,
                                const Vector2f* dImgCoords,
                                uint32_t sampleCount,
                                const CudaSystem& system) override;
        void        FilterToImg(ImageMemory&,
                                const float* dValues,
                                const Vector2f* dImgCoords,
                                uint32_t sampleCount,
                                const CudaSystem& system) override;
};

inline GPUMitchellFilterFunctor::GPUMitchellFilterFunctor(float radius, float b, float c)
    : radius(radius)
    , radiusRecip(Vector2f(1.0f) / radius)
    , coeffs01(0.0f)
    , coeffs12(0.0f)
{
    static constexpr float t = 1.0f / 6.0f;
    coeffs01[0] = t * ( 12 -  9 * b -  6 * c);
    coeffs01[1] = t * (-18 + 12 * b +  6 * c);
    coeffs01[2] = 0.0f;
    coeffs01[3] = t * (  6 +  2 * b +  0 * c);

    coeffs12[0] = t * (- 1 * b -  6 * c);
    coeffs12[1] = t * (  6 * b + 30 * c);
    coeffs12[2] = t * (-12 * b - 48 * c);
    coeffs12[3] = t * (  8 * b + 24 * c);
}

__device__ __host__ inline
float GPUMitchellFilterFunctor::Mitchell1D(float x) const
{
    x = fabs(x);
    float x2 = x * x;
    float x3 = x2 * x;

    Vector4f coeffs = Zero4f;
    if(x < 1)
        coeffs = coeffs01;
    else if(x < 2)
        coeffs = coeffs12;

    float result = (coeffs[0] * x3 +
                    coeffs[1] * x2 +
                    coeffs[2] * x +
                    coeffs[3]);
    return result;
}

__device__ __host__ inline
float GPUMitchellFilterFunctor::operator()(const Vector2f& pixCoord,
                                           const Vector2f& sampleCoord) const
{
    Vector2f dist = (pixCoord - sampleCoord).Abs();
    Vector2f mitchell2D = Vector2f(Mitchell1D(2.0f * dist[0] * radiusRecip[0]),
                                   Mitchell1D(2.0f * dist[1] * radiusRecip[1]));
    return mitchell2D.Multiply();
}

inline GPUReconFilterMitchell::GPUReconFilterMitchell(float filterRadius, Options filterOptions)
    : GPUReconFilter(filterRadius)
    , filter()
{
    float b, c;
    TracerError err = TracerError::OK;
    if((err = filterOptions.GetFloat(b, B_OPTION_NAME)) != TracerError::OK)
        throw TracerException(err, "Mitchell Filter parameter \"b\" not found");
    if((err = filterOptions.GetFloat(c, C_OPTION_NAME)) != TracerError::OK)
        throw TracerException(err, "Mitchell Filter parameter \"c\" not found");

    filter = GPUMitchellFilterFunctor(filterRadius, b, c);
}

inline
void GPUReconFilterMitchell::FilterToImg(ImageMemory& iMem,
                                         const Vector4f* dValues,
                                         const Vector2f* dImgCoords,
                                         uint32_t sampleCount,
                                         const CudaSystem& system)
{
    FilterToImgInternal(iMem, dValues, dImgCoords, sampleCount,
                        filter, system);
}

inline
void GPUReconFilterMitchell::FilterToImg(ImageMemory& iMem,
                                         const float* dValues,
                                         const Vector2f* dImgCoords,
                                         uint32_t sampleCount,
                                         const CudaSystem& system)
{
    FilterToImgInternal(iMem, dValues, dImgCoords, sampleCount,
                        filter, system);
}

static_assert(IsTracerClass<GPUReconFilterMitchell>::value,
              "\"GPUReconFilterMitchell\" is not a TracerClass!");