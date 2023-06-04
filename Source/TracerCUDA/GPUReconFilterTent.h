#pragma once

#include "GPUReconFilter.h"
#include "TypeTraits.h"

class GPUTentFilterFunctor
{
    private:
        Vector2f    radius;
        Vector2f    recipRadius;

    public:
        // Constructors & Destructor
                    GPUTentFilterFunctor(float radius);
                    ~GPUTentFilterFunctor() = default;

        __device__ __host__
        float       operator()(const Vector2f& pixCoord,
                               const Vector2f& sampleCoord) const;
};

class GPUReconFilterTent : public GPUReconFilter
{
    public:
        static const char*      TypeName() { return "Tent"; }
        using FilterFunctor     = GPUTentFilterFunctor;
    private:
        FilterFunctor           filter;

    protected:
    public:
        // Constructors & Destructor
                    GPUReconFilterTent(float filterRadius, Options filterOptions);
                    ~GPUReconFilterTent() = default;

        //
        const char* Type() const override { return TypeName(); }

        void        FilterToImg(ImageMemory&,
                                const Vector4f* dValues,
                                const Vector2f* dImgCoords,
                                uint32_t sampleCount,
                                const CudaSystem& system,
                                float scalarMultiplier = 1.0f) override;
        void        FilterToImg(ImageMemory&,
                                const float* dValues,
                                const Vector2f* dImgCoords,
                                uint32_t sampleCount,
                                const CudaSystem& system,
                                float scalarMultiplier = 1.0f) override;
};

inline GPUTentFilterFunctor::GPUTentFilterFunctor(float radius)
    : radius(radius)
    , recipRadius(1.0f / radius)
{}

__device__ __host__ inline
float GPUTentFilterFunctor::operator()(const Vector2f& pixCoord,
                                       const Vector2f& sampleCoord) const
{
    // Calculate from the distance
    // not it is [filterRadius, 0]
    Vector2f linearWeight = Vector2f::Max(Vector2f(0.0f),
                                          radius - (pixCoord - sampleCoord).Abs());
    // Now it is [1, 0]
    // Technically you don't need this but making
    // weights around 1 should be better for precision?
    linearWeight *= recipRadius;

    return linearWeight[0] * linearWeight[1];
}

inline GPUReconFilterTent::GPUReconFilterTent(float filterRadius, Options filterOptions)
    : GPUReconFilter(filterRadius)
    , filter(filterRadius)
{}

inline
void GPUReconFilterTent::FilterToImg(ImageMemory& iMem,
                                     const Vector4f* dValues,
                                     const Vector2f* dImgCoords,
                                     uint32_t sampleCount,
                                     const CudaSystem& system,
                                     float scalarMultiplier)
{
    FilterToImgInternal(iMem, dValues, dImgCoords, sampleCount, scalarMultiplier,
                        filter, system);
}

inline
void GPUReconFilterTent::FilterToImg(ImageMemory& iMem,
                                     const float* dValues,
                                     const Vector2f* dImgCoords,
                                     uint32_t sampleCount,
                                     const CudaSystem& system,
                                     float scalarMultiplier)
{
    FilterToImgInternal(iMem, dValues, dImgCoords, sampleCount, scalarMultiplier,
                        filter, system);
}

static_assert(IsTracerClass<GPUReconFilterTent>::value,
              "\"GPUReconFilterTent\" is not a TracerClass!");