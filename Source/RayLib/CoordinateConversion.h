#pragma once

#include "Vector.h"
#include "HybridFunctions.h"
#include <type_traits>

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

    // Cocentric Octohedral Mapping
    // https://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf
    template<class T>
    __host__ __device__ HYBRID_INLINE
    FloatEnable<T, Vector<2, T>> DirectionToCocentricOctohedral(const Vector<3, T>&);

    template<class T>
    __host__ __device__ HYBRID_INLINE
    FloatEnable<T, Vector<3, T>> CocentricOctohedralToDirection(const Vector<2, T>&);

    template<class T>
    __host__ __device__ HYBRID_INLINE
    FloatEnable<T, Vector<2, T>> CocentricOctohedralWrap(const Vector<2, T>&);

    template<class T>
    __host__ __device__ HYBRID_INLINE
    IntegralEnable<T, Vector<2, T>> CocentricOctohedralWrapInt(const Vector<2, T>& st,
                                                               const Vector<2, T>& dimensions);
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

template<class T>
__host__ __device__ HYBRID_INLINE
FloatEnable<T, Vector<2, T>> Utility::DirectionToCocentricOctohedral(const Vector<3, T>& dir)
{
    // Edge case
    if(dir[0] == 0 && dir[1] == 0) return Vector<2, T>(0);

    static constexpr T TwoOvrPi = static_cast<T>(MathConstants::InvPi_d * 2.0);

    T xAbs = abs(dir[0]);
    T yAbs = abs(dir[1]);
    T atanIn = yAbs / xAbs;
    T phiPrime = atan(atanIn);

    T radius = sqrt(1 - abs(dir[2]));

    T v = radius * TwoOvrPi * phiPrime;
    T u = radius - v;
    // Now convert to the quadrant
    if(dir[2] < 0)
    {
        T uPrime = 1.0f - v;
        T vPrime = 1.0f - u;
        u = uPrime;
        v = vPrime;
    }
    // Sign extend the uv
    u *= (signbit(dir[0]) ? -1 : 1);
    v *= (signbit(dir[1]) ? -1 : 1);

    // Finally
    // [-1,1] to [0,1]
    Vector<2, T> st = Vector<2, T>(u, v);
    st = (st + 1) * static_cast<T>(0.5);
    return st;
}

template<class T>
__host__ __device__ HYBRID_INLINE
FloatEnable<T, Vector<3, T>> Utility::CocentricOctohedralToDirection(const Vector<2, T>& st)
{
    static constexpr T piOvr4 = static_cast<T>(MathConstants::Pi_d * 0.25);

    // [0,1] to [-1,1]
    Vector<2, T> uv = st * 2 - 1;
    Vector<2, T> uvAbs = uv.Abs();

    // Radius
    T d = 1 - uvAbs.Sum();
    T radius = 1 - abs(d);
    T phiPrime = 0;
    // Avoid division by zero
    if(radius != 0) phiPrime = ((uvAbs[1] - uvAbs[0]) / radius + 1) * piOvr4;
    // Coords
    T cosPhi = (signbit(uv[0]) ? -1 : 1) * cos(phiPrime);
    T sinPhi = (signbit(uv[1]) ? -1 : 1) * sin(phiPrime);
    T z = (signbit(d) ? -1 : 1) * (1 - radius * radius);

    // Now all is OK do the cocentric disk stuff
    T xyFactor = radius * sqrt(2 - radius * radius);
    T x = cosPhi * xyFactor;
    T y = sinPhi * xyFactor;

    return Vector<3, T>(x, y, z);
}

template<class T>
__host__ __device__ HYBRID_INLINE
FloatEnable<T, Vector<2, T>> Utility::CocentricOctohedralWrap(const Vector<2, T>& st)
{
    // TODO: do more proper implementation
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
    using IntType = typename std::conditional_t<std::is_same_v<T, double>, int64_t, int32_t>;

    // Given st => (-inf, inf) convert to [0, 1]
    // Octohedral Cocentric mapping has straightforward properties
    // if either s or t is odd (integral part) we mirror the st on both sides
    // If both is odd or even do not mirror

    // Convert the negative numbers
    Vector2f stConv = st;
    if(st[0] < 0) stConv[0] = -2 - st[0];
    if(st[1] < 0) stConv[1] = -2 - st[1];

    T iS; T fS = abs(modf(stConv[0], &iS));
    T iT; T fT = abs(modf(stConv[1], &iT));
    IntType iSInt = static_cast<IntType>(iS);
    IntType iTInt = static_cast<IntType>(iT);
    bool doMirror = static_cast<bool>((iSInt & 0x1) ^ (iTInt & 0x1));
    if(doMirror)
    {
        fS = 1 - fS;
        fT = 1 - fT;
    }
    return Vector<2, T>(fS, fT);
}

template<class T>
__host__ __device__ HYBRID_INLINE
IntegralEnable<T, Vector<2, T>> Utility::CocentricOctohedralWrapInt(const Vector<2, T>& st,
                                                                    const Vector<2, T>& dimensions)
{
    Vector<2, T> stConv = st;
    if constexpr(std::is_signed_v<T>)
    {
        if(st[0] < 0) stConv[0] = -2 * dimensions[0] - st[0];
        if(st[1] < 0) stConv[1] = -2 * dimensions[1] - st[1];
    }

    Vector<2, T> dimClamp = dimensions - 1;
    Vector<2, T> intPart = stConv / dimensions;
    Vector<2, T> fracPart = (stConv % dimensions).Abs();

    T xOdd = (intPart[0] & 0x1);
    T yOdd = (intPart[1] & 0x1);
    bool doMirror = static_cast<bool>(xOdd ^ yOdd);
    if(doMirror) fracPart = dimClamp - fracPart;
    return fracPart;
}