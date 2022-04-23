#pragma once

#include "CudaCheck.h"
#include <algorithm>

namespace HybridFuncs
{
    template <class T>
    __device__ __host__ HYBRID_INLINE
    void   Swap(T&, T&);

    template <class T>
    __device__ __host__ HYBRID_INLINE
    T      Clamp(const T&, const T& min, const T& max);

    template <class T, class F>
    __device__ __host__ HYBRID_INLINE
    T      Lerp(const T& a, const T& b, const F& v);
}

template <class T>
__device__ __host__
void HybridFuncs::Swap(T& t0, T& t1)
{
    T temp = t0;
    t0 = t1;
    t1 = temp;
}

template <>
__device__ __host__ HYBRID_INLINE
float HybridFuncs::Clamp(const float& t, const float& minVal, const float& maxVal)
{
    return fmin(fmax(minVal, t), maxVal);
}

template <>
__device__ __host__ HYBRID_INLINE
double HybridFuncs::Clamp(const double& t, const double& minVal, const double& maxVal)
{
    return fmin(fmax(minVal, t), maxVal);
}

template <class T, class F>
__device__ __host__ HYBRID_INLINE
T HybridFuncs::Lerp(const T& a, const T& b, const F& v)
{
    assert(v >= 0.0 && v <= 1);
    return a * (1 - v) + b * v;
}