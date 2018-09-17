#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <type_traits>
#include "FunctionEnablers.cuh"

template<class Type>
using CompFunc = bool(*)(const Type&, const Type&);

template <class T>
inline __device__ EnableRest<T, bool> CompareST(const T& a, const T&b)
{
	return a < b;
}

template <class T>
inline __device__ EnableRest<T, bool> CompareGT(const T& a, const T&b)
{
	return a > b;
}

// Dim 1
template <class T>
inline __device__ EnableNV1Type<T, bool> CompareST(const T& a, const T&b)
{
	return a.x < b.x;
}

template <class T>
inline __device__ EnableNV1Type<T, bool> CompareGT(const T& a, const T&b)
{
	return a.x > b.x;
}

// Dim 2
template <class T>
inline __device__ EnableNV2Type<T, bool> ReduceST(const T& a, const T&b)
{
	return (a.x < b.x &&
			a.y < b.y);
}

template <class T>
inline __device__ EnableNV2Type<T, bool> ReduceGT(const T& a, const T&b)
{
	return (a.x > b.x &&
			a.y > b.y);
}

// Dim 3
template <class T>
inline __device__ EnableNV3Type<T, bool> ReduceST(const T& a, const T&b)
{
	return (a.x < b.x &&
			a.y < b.y &&
			a.z < b.z);
}

template <class T>
inline __device__ EnableNV3Type<T, bool> ReduceGT(const T& a, const T&b)
{
	return (a.x > b.x &&
			a.y > b.y &&
			a.z > b.z);
}

// Dim 4
template <class T>
inline __device__ EnableNV4Type<T, bool> ReduceST(const T& a, const T&b)
{
	return (a.x < b.x &&
			a.y < b.y &&
			a.z < b.z &&
			a.w < b.w);
}

template <class T>
inline __device__ EnableNV4Type<T, bool> ReduceGT(const T& a, const T&b)
{
	return (a.x > b.x &&
			a.y > b.y &&
			a.z > b.z &&
			a.w > b.w);
}

// We have to wrap out functions to Functors
// Kernels give invalid program counter
// It may mean wrong (enable_if disabled) function is  propogated?
// Wrapping prevent that
template<class Type, CompFunc<Type> F>
struct CompareWrapper
{
	__host__ __device__ __forceinline__
	bool operator()(const Type &a, const Type &b) const
	{
		return F(a, b);
	}
};