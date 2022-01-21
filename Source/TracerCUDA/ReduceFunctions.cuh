#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "FunctionEnablers.cuh"

template<class Type>
using ReduceFunc = Type(*)(const Type&, const Type&);

template <class T>
__device__ inline
EnableRest<T, T> ReduceAdd(const T& a, const T&b)
{
    return a + b;
}

template <class T>
__device__ inline
EnableRest<T, T> ReduceSubtract(const T& a, const T&b)
{
    return a - b;
}

template <class T>
__device__ inline
EnableRest<T, T> ReduceMultiply(const T& a, const T&b)
{
    return a * b;
}

template <class T>
__device__ inline
EnableRest<T, T> ReduceDivide(const T& a, const T&b)
{
    return a / b;
}

template<class T>
__device__ inline
EnableArithmetic<T, T> ReduceMin(const T& a, const T&b)
{
    return min(a, b);
}

template<class T>
__device__ inline
EnableArithmetic<T, T> ReduceMax(const T& a, const T&b)
{
    return max(a, b);
}

template<class T>
__device__ inline
EnableVectorOrMatrix<T, T> ReduceMin(const T& a, const T&b)
{
    return T::Min(a, b);
}

template<class T>
__device__ inline
EnableVectorOrMatrix<T, T> ReduceMax(const T& a, const T&b)
{
    return T::Max(a, b);
}

// Dim 1
template <class T>
__device__ inline
EnableNV1Type<T, T> ReduceAdd(const T& a, const T&b)
{
    return T{a.x + b.x};
}

template <class T>
__device__ inline
EnableNV1Type<T, T> ReduceSubtract(const T& a, const T&b)
{
    return T{a.x - b.x};
}

template <class T>
__device__ inline
EnableNV1Type<T, T> ReduceMultiply(const T& a, const T&b)
{
    return T{a.x * b.x};
}

template <class T>
__device__ inline
EnableNV1Type<T, T> ReduceDivide(const T& a, const T&b)
{
    return T{a.x / b.x};
}

template<class T>
__device__ inline
EnableNV1Type<T, T> ReduceMin(const T& a, const T&b)
{
    return T{min(a.x, b.x)};
}

template<class T>
__device__ inline
EnableNV1Type<T, T> ReduceMax(const T& a, const T&b)
{
    return T{max(a.x, b.x)};
}

// Dim 2
template <class T>
__device__ inline
EnableNV2Type<T, T> ReduceAdd(const T& a, const T&b)
{
    return T
    {
        a.x + b.x,
        a.y + b.y
    };
}

template <class T>
__device__ inline
EnableNV2Type<T, T> ReduceSubtract(const T& a, const T&b)
{
    return T
    {
        a.x - b.x,
        a.y - b.y
    };
}

template <class T>
__device__ inline
EnableNV2Type<T, T> ReduceMultiply(const T& a, const T&b)
{
    return T
    {
        a.x * b.x,
        a.y * b.y
    };
}

template <class T>
__device__ inline
EnableNV2Type<T, T> ReduceDivide(const T& a, const T&b)
{
    return T
    {
        a.x / b.x,
        a.y / b.y
    };
}

template<class T>
__device__ inline
EnableNV2Type<T, T> ReduceMin(const T& a, const T&b)
{
    return T
    {
        min(a.x, b.x),
        min(a.y, b.y)
    };
}

template<class T>
__device__ inline
EnableNV2Type<T, T> ReduceMax(const T& a, const T&b)
{
    return T
    {
        max(a.x, b.x),
        max(a.y, b.y)
    };
}

// Dim 3
template <class T>
__device__ inline
EnableNV3Type<T, T> ReduceAdd(const T& a, const T&b)
{
    return T
    {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}

template <class T>
__device__ inline
EnableNV3Type<T, T> ReduceSubtract(const T& a, const T&b)
{
    return T
    {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
}

template <class T>
__device__ inline
EnableNV3Type<T, T> ReduceMultiply(const T& a, const T&b)
{
    return T
    {
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    };
}

template <class T>
__device__ inline
EnableNV3Type<T, T> ReduceDivide(const T& a, const T&b)
{
    return T
    {
        a.x / b.x,
        a.y / b.y,
        a.z / b.z
    };
}

template<class T>
__device__ inline
EnableNV3Type<T, T> ReduceMin(const T& a, const T&b)
{
    return T
    {
        min(a.x, b.x),
        min(a.y, b.y),
        min(a.z, b.z),
    };
}

template<class T>
__device__ inline
EnableNV3Type<T, T> ReduceMax(const T& a, const T&b)
{
    return T
    {
        max(a.x, b.x),
        max(a.y, b.y),
        max(a.z, b.z)
    };
}

// Dim 4
template <class T>
__device__ inline
EnableNV4Type<T, T> ReduceAdd(const T& a, const T&b)
{
    return T
    {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
        a.w + b.w
    };
}

template <class T>
__device__ inline
EnableNV4Type<T, T> ReduceSubtract(const T& a, const T&b)
{
    return T
    {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
        a.w - b.w
    };
}

template <class T>
__device__ inline
EnableNV4Type<T, T> ReduceMultiply(const T& a, const T&b)
{
    return T
    {
        a.x * b.x,
        a.y * b.y,
        a.z * b.z,
        a.w * b.w
    };
}

template <class T>
__device__ inline
EnableNV4Type<T, T> ReduceDivide(const T& a, const T&b)
{
    return T
    {
        a.x / b.x,
        a.y / b.y,
        a.z / b.z,
        a.w / b.w
    };
}

template<class T>
__device__ inline
EnableNV4Type<T, T> ReduceMin(const T& a, const T&b)
{
    return T
    {
        min(a.x, b.x),
        min(a.y, b.y),
        min(a.z, b.z),
        min(a.w, b.w),
    };
}

template<class T>
__device__ inline
EnableNV4Type<T, T> ReduceMax(const T& a, const T&b)
{
    return T
    {
        max(a.x, b.x),
        max(a.y, b.y),
        min(a.z, b.z),
        max(a.w, b.w),
    };
}

// We have to wrap out functions to Functors
// Kernels give invalid program counter
// It may mean wrong (enable_if disabled) function is  propagated?
// Wrapping prevent that
template<class Type, ReduceFunc<Type> F>
struct ReduceWrapper
{
    __host__ __device__ HYBRID_INLINE
    Type operator()(const Type &a, const Type &b) const
    {
        return F(a, b);
    }
};