#pragma once

#include "RayLib/Vector.h"

namespace MortonCode
{
    namespace Detail
    {
        template<class T>
        __device__ __host__ HYBRID_INLINE
        T           Expand3D(uint32_t);

        template<class T>
        __device__ __host__ HYBRID_INLINE
        uint32_t    Shrink3D(T x);

        template<class T>
        __device__ __host__ HYBRID_INLINE
        T           Expand2D(uint32_t);

        template<class T>
        __device__ __host__ HYBRID_INLINE
        uint32_t    Shrink2D(T x);
    }

    template <class T>
    __device__ __host__ HYBRID_INLINE
    T           Compose3D(const Vector3ui&);

    template <class T>
    __device__ __host__ HYBRID_INLINE
    Vector3ui   Decompose3D(T code);

    template <class T>
    __device__ __host__ HYBRID_INLINE
    T           Compose2D(const Vector2ui&);

    template <class T>
    __device__ __host__ HYBRID_INLINE
    Vector2ui   Decompose2D(T code);
}

template<>
__device__ __host__ inline
uint64_t MortonCode::Detail::Expand3D(uint32_t val)
{
    // https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit
    uint64_t x = val;
    x &= 0x1fffff;
    x = (x | x << 32) & 0x001f00000000ffff;
    x = (x | x << 16) & 0x001f0000ff0000ff;
    x = (x | x << 8 ) & 0x100f00f00f00f00f;
    x = (x | x << 4 ) & 0x10c30c30c30c30c3;
    x = (x | x << 2 ) & 0x1249249249249249;
    return x;
}

template<>
__device__ __host__ inline
uint32_t MortonCode::Detail::Expand3D(uint32_t val)
{
    // https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit
    uint32_t x = val;
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8 ) & 0x300f00f;
    x = (x | x << 4 ) & 0x30c30c3;
    x = (x | x << 2 ) & 0x9249249;
    return x;
}

template<>
__device__ __host__ inline
uint32_t MortonCode::Detail::Shrink3D(uint32_t x)
{
    x = x & 0x55555555;
    x = (x | x >> 1) & 0x13333333;
    x = (x | x >> 2) & 0x0F0F0F0F;
    x = (x | x >> 4) & 0x00FF00FF;
    x = (x | x >> 8) & 0x0000FFFF;
    return x;
}

template<>
__device__ __host__ inline
uint32_t MortonCode::Detail::Shrink3D(uint64_t x)
{
    x &=                  0x1249249249249249;
    x = (x ^ (x >> 2))  & 0x30c30c30c30c30c3;
    x = (x ^ (x >> 4))  & 0xf00f00f00f00f00f;
    x = (x ^ (x >> 8))  & 0x00ff0000ff0000ff;
    x = (x ^ (x >> 16)) & 0x00ff00000000ffff;
    x = (x ^ (x >> 32)) & 0x00000000001fffff;
    return static_cast<uint32_t>(x);
}

template<>
__device__ __host__ inline
uint64_t MortonCode::Detail::Expand2D(uint32_t val)
{
    // https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
    uint64_t x = val;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x <<  8)) & 0x00FF00FF00FF00FF;
    x = (x | (x <<  4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x <<  2)) & 0x3333333333333333;
    x = (x | (x <<  1)) & 0x5555555555555555;
    return x;
}

template<>
__device__ __host__ inline
uint32_t MortonCode::Detail::Expand2D(uint32_t val)
{
    // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
    uint32_t x = val;
    x &= 0x0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f;
    x = (x ^ (x << 2)) & 0x33333333;
    x = (x ^ (x << 1)) & 0x55555555;
    return x;
}

template<>
__device__ __host__ inline
uint32_t MortonCode::Detail::Shrink2D(uint32_t x)
{
    x &= 0x55555555;
    x = (x ^ (x >> 1)) & 0x33333333;
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    x = (x ^ (x >> 8)) & 0x0000ffff;
    return x;
}

template<>
__device__ __host__ inline
uint32_t MortonCode::Detail::Shrink2D(uint64_t x)
{
    // https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
    x = x               & 0x5555555555555555;
    x = (x | (x >>  1)) & 0x3333333333333333;
    x = (x | (x >>  2)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >>  4)) & 0x00FF00FF00FF00FF;
    x = (x | (x >>  8)) & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
    return static_cast<uint32_t>(x);
}

template <class T>
__device__ __host__ inline
T MortonCode::Compose3D(const Vector3ui& val)
{
    T x = Detail::Expand3D<T>(val[0]);
    T y = Detail::Expand3D<T>(val[1]);
    T z = Detail::Expand3D<T>(val[2]);
    return ((x << 0) | (y << 1) | (z << 2));
}

template <class T>
__device__ __host__ inline
Vector3ui MortonCode::Decompose3D(T code)
{
    T x = Detail::Shrink3D<T>(code >> 0);
    T y = Detail::Shrink3D<T>(code >> 1);
    T z = Detail::Shrink3D<T>(code >> 2);
    return Vector3ui(x, y, z);
}

template <class T>
__device__ __host__ inline
T MortonCode::Compose2D(const Vector2ui& val)
{
    T x = Detail::Expand2D<T>(val[0]);
    T y = Detail::Expand2D<T>(val[1]);
    return ((x << 0) | (y << 1));
}

template <class T>
__device__ __host__ inline
Vector2ui MortonCode::Decompose2D(T code)
{
    T x = Detail::Shrink2D<T>(code >> 0);
    T y = Detail::Shrink2D<T>(code >> 1);
    return Vector2ui(x, y);
}