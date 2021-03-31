#pragma once

#include "Vector.h"

namespace Utility
{
    template<class T>
    __host__ __device__
    FloatEnable<T, Vector<3, T>> SphericalToCartesian(const Vector<3, T>&);

    template<class T>
    __host__ __device__
    FloatEnable<T, Vector<3, T>> CartesianToSpherical(const Vector<3, T>&);

    template<class T>
    __host__ __device__
    FloatEnable<T, Vector<3, T>> SphericalToCartesianUnit(const Vector<2, T>&);

    template<class T>
    __host__ __device__
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
    T r = cart.Length();
    // range [-pi, pi]
    T azimuth = atan2(cart[1], cart[0]);
    // range [0, pi]
    T incl = acos(cart[2]);
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
FloatEnable<T, Vector<2, T>> Utility::CartesianToSphericalUnit(const Vector<3, T>& cart)
{
    // Convert to Spherical Coordinates
    // range [-pi, pi]
    T azimuth = atan2(cart[1], cart[0]);
    // range [0, pi]
    T incl = acos(cart[2]);

    return Vector<2, T>(azimuth, incl);
}