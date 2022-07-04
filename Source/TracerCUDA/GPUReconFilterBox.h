#pragma once

#include "GPUReconFilter.h"
#include "TypeTraits.h"

class GPUBoxFilterFunctor
{
    private:
        float       radius;

    public:
        // Constructors & Destructor
                    GPUBoxFilterFunctor(float radius);
                    ~GPUBoxFilterFunctor() = default;

        __device__ __host__
        float       operator()(const Vector2f& pixCoord,
                               const Vector2f& sampleCoord) const;
};

class GPUReconFilterBox : public GPUReconFilter
{
    public:
        static const char* TypeName() { return "Box"; }
    private:
        GPUBoxFilterFunctor   filter;

    protected:
    public:
        // Constructors & Destructor
                    GPUReconFilterBox(float filterRadius, Options filterOptions);
                    ~GPUReconFilterBox() = default;

        //
        const char* Type() const override { return TypeName(); }

        void        FilterToImg(ImageMemory&,
                                const Vector4f* dValues,
                                const Vector2f* dImgCoords,
                                uint32_t sampleCount,
                                const CudaSystem& system) override;
};

inline GPUBoxFilterFunctor::GPUBoxFilterFunctor(float radius)
    : radius(radius)
{}

__device__ __host__ inline
float GPUBoxFilterFunctor::operator()(const Vector2f& pixCoord,
                                      const Vector2f& sampleCoord) const
{
    return ((pixCoord - sampleCoord).Length() < radius) ? 1.0f : 0.0f;
}

inline GPUReconFilterBox::GPUReconFilterBox(float filterRadius, Options filterOptions)
    : GPUReconFilter(filterRadius)
    , filter(filterRadius)
{}

inline
void GPUReconFilterBox::FilterToImg(ImageMemory& iMem,
                                    const Vector4f* dValues,
                                    const Vector2f* dImgCoords,
                                    uint32_t sampleCount,
                                    const CudaSystem& system)
{
    FilterToImgInternal(iMem, dValues, dImgCoords, sampleCount,
                        filter, system);
}

static_assert(IsTracerClass<GPUReconFilterBox>::value,
              "\"GPUReconFilterBox\" is not a TracerClass!");