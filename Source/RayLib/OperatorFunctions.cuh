#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <type_traits>
#include "Vector.cuh"
#include "Matrix.h"
#include "Quaternion.h"

#define ENABLE_NV_TYPE(count) \
		std::is_same<T, float##count>::value		|| \
		std::is_same<T, double##count>::value		|| \
		std::is_same<T, short##count>::value		|| \
		std::is_same<T, ushort##count>::value		|| \
		std::is_same<T, int##count>::value			|| \
		std::is_same<T, uint##count>::value			|| \
		std::is_same<T, long##count>::value			|| \
		std::is_same<T, ulong##count>::value		|| \
		std::is_same<T, long##count>::value			|| \
		std::is_same<T, ulong##count>::value

// Enable Ifs
template<class T>
using EnableNV1Type = typename std::enable_if<ENABLE_NV_TYPE(1), T>::type;

//template<class T>
//using EnableNV2Type = typename std::enable_if<ENABLE_NV_TYPE(2), T>::type;

template<class T>
using EnableNV2Type = typename std::enable_if<std::is_same<T, float2>::value, T>::type;

template<class T>
using EnableNV3Type = typename std::enable_if<ENABLE_NV_TYPE(3), T>::type;

template<class T>
using EnableNV4Type = typename std::enable_if<ENABLE_NV_TYPE(4), T>::type;

template<class T>
using EnableArithmetic = typename std::enable_if<std::is_arithmetic<T>::value, T>::type;

template<class T>
using EnableVectorOrMatrix = typename std::enable_if<IsVectorType<T>::value ||
													 IsMatrixType<T>::value, T>::type;

template<class T>
using EnableRest = typename std::enable_if<IsVectorType<T>::value ||
										   IsMatrixType<T>::value || 
										   IsQuatType<T>::value ||
										   std::is_arithmetic<T>::value, T>::type;

template<class Type>
using ReduceFunc = Type(*)(const Type&, const Type&);

template <class T>
inline __device__ EnableRest<T> ReduceAdd(const T& a, const T&b)
{
	return a + b;
}

template <class T>
inline __device__ EnableRest<T> ReduceSubtract(const T& a, const T&b)
{
	return a - b;
}

template <class T>
inline __device__ EnableRest<T> ReduceMultiply(const T& a, const T&b)
{
	return a * b;
}

template <class T>
inline __device__ EnableRest<T> ReduceDivide(const T& a, const T&b)
{
	return a / b;
}

template<class T>
inline __device__  EnableArithmetic<T> ReduceMin(const T& a, const T&b)
{
	return std::min(a, b);
}

template<class T>
inline __device__ EnableArithmetic<T> ReduceMax(const T& a, const T&b)
{
	return std::max(a, b);
}

template<class T>
inline __device__  EnableVectorOrMatrix<T> ReduceMin(const T& a, const T&b)
{
	return T::Min(a, b);
}

template<class T>
inline __device__  EnableVectorOrMatrix<T> ReduceMax(const T& a, const T&b)
{
	return T::Max(a, b);
}

// Dim 1
template <class T>
inline __device__ EnableNV1Type<T> ReduceAdd(const T& a, const T&b)
{
	return T{a.x + b.x};
}

template <class T>
inline __device__ EnableNV1Type<T> ReduceSubtract(const T& a, const T&b)
{
	return T{a.x - b.x};
}

template <class T>
inline __device__ EnableNV1Type<T> ReduceMultiply(const T& a, const T&b)
{
	return T{a.x * b.x};
}

template <class T>
inline __device__ EnableNV1Type<T> ReduceDivide(const T& a, const T&b)
{
	return T{a.x / b.x};
}

template<class T>
inline __device__  EnableNV1Type<T> ReduceMin(const T& a, const T&b)
{
	return T{std::min(a.x, b.x)};
}

template<class T>
inline __device__ EnableNV1Type<T> ReduceMax(const T& a, const T&b)
{
	return T{std::max(a.x, b.x)};
}

// Dim 2
template <class T>
inline __device__ EnableNV2Type<T> ReduceAdd(const T& a, const T&b)
{
	return T
	{
		a.x + b.x,
		a.y + b.y
	};
}

template <class T>
inline __device__ EnableNV2Type<T> ReduceSubtract(const T& a, const T&b)
{
	return T
	{
		a.x - b.x,
		a.y - b.y
	};
}

template <class T>
inline __device__ EnableNV2Type<T> ReduceMultiply(const T& a, const T&b)
{
	return T
	{
		a.x * b.x,
		a.y * b.y
	};
}

template <class T>
inline __device__ EnableNV2Type<T> ReduceDivide(const T& a, const T&b)
{
	return T
	{
		a.x / b.x,
		a.y / b.y
	}; 
}

template<class T>
inline __device__  EnableNV2Type<T> ReduceMin(const T& a, const T&b)
{
	return T
	{
		std::min(a.x, b.x),
		std::min(a.y, b.y)
	};
}

template<class T>
inline __device__ EnableNV2Type<T> ReduceMax(const T& a, const T&b)
{
	return T
	{
		std::max(a.x, b.x),
		std::max(a.y, b.y)
	};
}

// Dim 3
template <class T>
inline __device__ EnableNV3Type<T> ReduceAdd(const T& a, const T&b)
{
	return T
	{
		a.x + b.x,
		a.y + b.y,
		a.z + b.z
	};
}

template <class T>
inline __device__ EnableNV3Type<T> ReduceSubtract(const T& a, const T&b)
{
	return T
	{
		a.x - b.x,
		a.y - b.y,
		a.z - b.z
	};
}

template <class T>
inline __device__ EnableNV3Type<T> ReduceMultiply(const T& a, const T&b)
{
	return T
	{
		a.x * b.x,
		a.y * b.y,
		a.z * b.z
	};
}

template <class T>
inline __device__ EnableNV3Type<T> ReduceDivide(const T& a, const T&b)
{
	return T
	{
		a.x / b.x,
		a.y / b.y,
		a.z / b.z
	};
}

template<class T>
inline __device__  EnableNV3Type<T> ReduceMin(const T& a, const T&b)
{
	return T
	{
		std::min(a.x, b.x),
		std::min(a.y, b.y),
		std::min(a.z, b.z),
	};
}

template<class T>
inline __device__ EnableNV3Type<T> ReduceMax(const T& a, const T&b)
{
	return T
	{
		std::max(a.x, b.x),
		std::max(a.y, b.y),
		std::max(a.z, b.z)
	};
}

// Dim 4
template <class T>
inline __device__ EnableNV4Type<T> ReduceAdd(const T& a, const T&b)
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
inline __device__ EnableNV4Type<T> ReduceSubtract(const T& a, const T&b)
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
inline __device__ EnableNV4Type<T> ReduceMultiply(const T& a, const T&b)
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
inline __device__ EnableNV4Type<T> ReduceDivide(const T& a, const T&b)
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
inline __device__  EnableNV4Type<T> ReduceMin(const T& a, const T&b)
{
	return T
	{
		std::min(a.x, b.x),
		std::min(a.y, b.y),
		std::min(a.z, b.z),
		std::min(a.w, b.w),
	};
}

template<class T>
inline __device__ EnableNV4Type<T> ReduceMax(const T& a, const T&b)
{
	return T
	{
		std::max(a.x, b.x),
		std::max(a.y, b.y),
		std::min(a.z, b.z),
		std::max(a.w, b.w),
	};
}