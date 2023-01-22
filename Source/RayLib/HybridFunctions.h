#pragma once

#include "CudaCheck.h"
#include <algorithm>

namespace HybridFuncs
{
    template <class T>
    HYBRD_FUNC void   Swap(T&, T&);
    template <class T>
    HYBRD_FUNC T      Clamp(const T&, const T& min, const T& max);
    template <class T, class F>
    HYBRD_FUNC T      Lerp(const T& a, const T& b, const F& v);
}

template <class T>
HYBRD_FUNC HYBRID_INLINE
void HybridFuncs::Swap(T& t0, T& t1)
{
    T temp = t0;
    t0 = t1;
    t1 = temp;
}

template <>
HYBRD_FUNC HYBRID_INLINE
float HybridFuncs::Clamp(const float& t, const float& minVal, const float& maxVal)
{
    return fmin(fmax(minVal, t), maxVal);
}

template <>
HYBRD_FUNC HYBRID_INLINE
double HybridFuncs::Clamp(const double& t, const double& minVal, const double& maxVal)
{
    return fmin(fmax(minVal, t), maxVal);
}

template <class T>
HYBRD_FUNC HYBRID_INLINE
T HybridFuncs::Clamp(const T& val, const T& min, const T& max)
{
    return (val < min) ? min : ((val > max) ? max : val);
}

template <class T, class F>
HYBRD_FUNC HYBRID_INLINE
T HybridFuncs::Lerp(const T& a, const T& b, const F& v)
{
    assert(v <= T(1) && v >= T(0));
    return a * (T(1) - v) + b * v;
}