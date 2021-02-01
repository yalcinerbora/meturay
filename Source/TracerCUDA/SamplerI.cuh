#pragma once

#include "RayLib/Vector.h"

// Texture Sampler Interface
template <int D, int C, class T>
class SamplerI
{
    public:
        virtual                             ~SamplerI() = default;

        // Interface
        virtual __device__ Vector<C, T>     Sample(const Vector<D, float>&) const = 0;
};


// Simple Texture with Constant Data Fallback 
template <int D, int C, class T>
class SamplerWithConstantFallback
{
    private:
        Vector<C, T>                constant;
        const SamplerI<D, C, T>*    dSampler;

    public:
        // Constructors & Destructor
                                    SamplerWithConstantFallback(const Vector<C, T>& data,
                                                                const SamplerI<D, C, T>* dSampler);
                                    ~SamplerWithConstantFallback() = default;
        
        // Interface
        __device__ 
        Vector<C, T>                Sample(const Vector<D, float>&) const;
};

template <int D, int C, class T>
SamplerWithConstantFallback<D,C,T>::SamplerWithConstantFallback(const Vector<C, T>& data,
                                                                const SamplerI<D, C, T>* dSampler)
    : constant(data)
    , dSampler(dSampler)
{}

template <int D, int C, class T>
__device__
inline Vector<C, T> SamplerWithConstantFallback<D, C, T>::Sample(const Vector<D, float>& v) const
{
    // If texture is available use it
    if(dSampler) return dSampler->Sample(v);
    // Or use the constant data
    else return constant;
}

template <int D, int C, class T>
using SamplerCF = SamplerWithConstantFallback<D, C, T>;