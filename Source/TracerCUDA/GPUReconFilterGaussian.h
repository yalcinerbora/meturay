#pragma once

#include "GPUReconFilter.h"
#include "TypeTraits.h"

#include "RayLib/TracerError.h"

// Gaussian filter reconstruction implementation
// from PBRT book

class GPUGaussianFilterFunctor
{
    private:
        float           negAlpha;
        Vector2f        radius;
        Vector2f        exponent;

        __device__ __host__
        float           Gauss(float x) const;

    public:
        // Constructors & Destructor
                        GPUGaussianFilterFunctor() = default;
                        GPUGaussianFilterFunctor(float radius, float alpha);
                        ~GPUGaussianFilterFunctor() = default;

        __device__ __host__
        float           operator()(const Vector2f& pixCoord,
                                   const Vector2f& sampleCoord) const;
};

class GPUReconFilterGaussian : public GPUReconFilter
{
    public:
        static const char*              TypeName() { return "Gaussian"; }
        using FilterFunctor             = GPUGaussianFilterFunctor;

        static constexpr const char*    ALPHA_OPTION_NAME = "alpha";
    private:
        FilterFunctor                   filter;

    protected:
    public:
        // Constructors & Destructor
                    GPUReconFilterGaussian(float filterRadius, Options filterOptions);
                    ~GPUReconFilterGaussian() = default;

        //
        const char* Type() const override { return TypeName(); }

        void        FilterToImg(ImageMemory&,
                                const Vector4f* dValues,
                                const Vector2f* dImgCoords,
                                uint32_t sampleCount,
                                const CudaSystem& system) override;
};

inline GPUGaussianFilterFunctor::GPUGaussianFilterFunctor(float radius, float alpha)
    : negAlpha(-alpha)
    , radius(radius)
    , exponent(Gauss(radius), Gauss(radius))
{}

__device__ __host__ inline
float GPUGaussianFilterFunctor::Gauss(float x) const
{
    return expf(negAlpha * x * x);
}

__device__ __host__ inline
float GPUGaussianFilterFunctor::operator()(const Vector2f& pixCoord,
                                           const Vector2f& sampleCoord) const
{
    // Clang min/max etc. definitions are only on std namespace
    // for host we need to do this
    // TODO: Is there a better solution?
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    Vector2f dist = (pixCoord - sampleCoord).Abs();
    Vector2f gauss2D = Vector2f(max(0.0f, Gauss(dist[0]) - exponent[0]),
                                max(0.0f, Gauss(dist[1]) - exponent[1]));
    return gauss2D.Multiply();
}

inline GPUReconFilterGaussian::GPUReconFilterGaussian(float filterRadius, Options filterOptions)
    : GPUReconFilter(filterRadius)
    , filter()
{
    float alpha;
    TracerError err = TracerError::OK;
    if((err = filterOptions.GetFloat(alpha, ALPHA_OPTION_NAME)) != TracerError::OK)
        throw TracerException(err, "Gauss Filter parameter \"alpha\" not found");

    filter = GPUGaussianFilterFunctor(filterRadius, alpha);
}

inline
void GPUReconFilterGaussian::FilterToImg(ImageMemory& iMem,
                                         const Vector4f* dValues,
                                         const Vector2f* dImgCoords,
                                         uint32_t sampleCount,
                                         const CudaSystem& system)
{
    FilterToImgInternal(iMem, dValues, dImgCoords, sampleCount,
                        filter, system);
}

static_assert(IsTracerClass<GPUReconFilterGaussian>::value,
              "\"GPUReconFilterGaussian\" is not a TracerClass!");