#pragma once

#include "Vector.h"
#include "HybridFunctions.h"

namespace Utility
{
    template<class T>
    __host__ __device__ HYBRID_INLINE
    FloatEnable<T, Vector<3, T>> SphericalToCartesian(const Vector<3, T>&);

    template<class T>
    __host__ __device__ HYBRID_INLINE
    FloatEnable<T, Vector<3, T>> CartesianToSpherical(const Vector<3, T>&);

    template<class T>
    __host__ __device__ HYBRID_INLINE
    FloatEnable<T, Vector<3, T>> SphericalToCartesianUnit(const Vector<2, T>&);

    template<class T>
    __host__ __device__ HYBRID_INLINE
    FloatEnable<T, Vector<3, T>> SphericalToCartesianUnit(const Vector<2, T>& sinCosTheta,
                                                          const Vector<2, T>& sinCosPhi);

    template<class T>
    __host__ __device__ HYBRID_INLINE
    FloatEnable<T, Vector<2, T>> CartesianToSphericalUnit(const Vector<3, T>&);
}

template<class T>
__host__ __device__
FloatEnable<T, Vector<3, T>> Utility::SphericalToCartesian(const Vector<3, T>& sphr)
{
    T x = sphr[0] * cos(sphr[1]) * sin(sphr[2]);
    T y = sphr[0] * sin(sphr[1]) * sin(sphr[2]);
    T z = sphr[0] * cos(sphr[2]);
    return Vector<3, T>(x, y, z);
}

template<class T>
__host__ __device__
FloatEnable<T, Vector<3, T>> Utility::CartesianToSpherical(const Vector<3, T>& cart)
{
    // Convert to Spherical Coordinates
    Vector<3, T> norm = cart.Normalize();
    T r = cart.Length();
    // range [-pi, pi]
    T azimuth = atan2(norm[1], norm[0]);
    // range [0, pi]
    T incl = acos(norm[2]);
    return Vector<3, T>(r, azimuth, incl);
}

template<class T>
__host__ __device__
FloatEnable<T, Vector<3, T>> Utility::SphericalToCartesianUnit(const Vector<2, T>& sphr)
{
    T x = cos(sphr[0]) * sin(sphr[1]);
    T y = sin(sphr[0]) * sin(sphr[1]);
    T z = cos(sphr[1]);
    return Vector<3, T>(x, y, z);
}

template<class T>
__host__ __device__
FloatEnable<T, Vector<3, T>> Utility::SphericalToCartesianUnit(const Vector<2, T>& sinCosTheta,
                                                               const Vector<2, T>& sinCosTPhi)
{
    T x = sinCosTheta[1] * sinCosTPhi[0];
    T y = sinCosTheta[0] * sinCosTPhi[0];
    T z = sinCosTPhi[1];
    return Vector<3, T>(x, y, z);
}

template<class T>
__host__ __device__
FloatEnable<T, Vector<2, T>> Utility::CartesianToSphericalUnit(const Vector<3, T>& cart)
{
    // Convert to Spherical Coordinates
    // range [-pi, pi]
    T azimuth = atan2(cart[1], cart[0]);
    // range [0, pi]
    // Sometimes normalized cartesian coords may invoke NaN here
    // clamp it to the range
    T incl = acos(HybridFuncs::Clamp<T>(cart[2], -1, 1));

    return Vector<2, T>(azimuth, incl);
}