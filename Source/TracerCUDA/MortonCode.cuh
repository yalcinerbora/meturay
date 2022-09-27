#pragma once

#include "RayLib/Vector.h"

namespace MortonCode
{
    namespace Detail
    {
        template<class T>
        __device__ __host__
        T           Expand(uint32_t);

        template<class T>
        __device__ __host__
        uint32_t    Shrink(T x);
    }

    template <class T>
    __device__ __host__
    T           Compose(const Vector3ui&);

    template <class T>
    __device__ __host__
    Vector3ui   Decompose(T code);
}

template<>
__device__ __host__ inline
uint64_t MortonCode::Detail::Expand(uint32_t val)
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
uint32_t MortonCode::Detail::Expand(uint32_t val)
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
uint32_t MortonCode::Detail::Shrink(uint32_t x)
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
uint32_t MortonCode::Detail::Shrink(uint64_t x)
{
    //x = x             & 0x5555555555555555;
    //x = (x | x >> 1 ) & 0x3333333333333333;
    //x = (x | x >> 2 ) & 0x0F0F0F0F0F0F0F0F;
    //x = (x | x >> 4 ) & 0x00FF00FF00FF00FF;
    //x = (x | x >> 8 ) & 0x0000FFFF0000FFFF;
    //x = (x | x >> 16) & 0x00000000FFFFFFFF;
    x &=                  0x1249249249249249;
    x = (x ^ (x >> 2))  & 0x30c30c30c30c30c3;
    x = (x ^ (x >> 4))  & 0xf00f00f00f00f00f;
    x = (x ^ (x >> 8))  & 0x00ff0000ff0000ff;
    x = (x ^ (x >> 16)) & 0x00ff00000000ffff;
    x = (x ^ (x >> 32)) & 0x00000000001fffff;
    return static_cast<uint32_t>(x);
}

template <class T>
__device__ __host__ inline
T MortonCode::Compose(const Vector3ui& val)
{
    T x = Detail::Expand<T>(val[0]);
    T y = Detail::Expand<T>(val[1]);
    T z = Detail::Expand<T>(val[2]);
    return ((x << 0) | (y << 1) | (z << 2));
}

template <class T>
__device__ __host__ inline
Vector3ui MortonCode::Decompose(T code)
{
    T x = Detail::Shrink<T>(code >> 0);
    T y = Detail::Shrink<T>(code >> 1);
    T z = Detail::Shrink<T>(code >> 2);
    return Vector3ui(x, y, z);
}