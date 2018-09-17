#pragma once
/**
Parallel Scan Algorithms
Can define custom reduce function and custom type

Moderen Reduction using Kepler warp data transfer
http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

Implementation of
https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

Nevermind. Currently found cub backend which what i was planning to do
Wrapping functions to it from now on.

*/

#include <cub/cub.cuh>

#include "DeviceMemory.h"
#include "CudaCheck.h"
#include "CudaConstants.h"
#include "ReduceFunctions.cuh"

template<class Type, ReduceFunc<Type> F>
__host__ void KCExclusiveScanArray(Type* out, const Type* in, 
								   size_t elementCount, Type identityElement,
								   cudaStream_t stream = (cudaStream_t)0)
{
	// Delegating to cub here
	size_t bufferSize = 0;
	cub::DeviceScan::ExclusiveScan(nullptr, bufferSize,
								   in, out,
								   ReduceWrapper<Type, F>(),
								   identityElement,
								   static_cast<int>(elementCount),
								   stream);

	DeviceMemory buffer(bufferSize);
	cub::DeviceScan::ExclusiveScan(buffer, bufferSize,
								   in, out,
								   ReduceWrapper<Type, F>(),
								   identityElement,
								   static_cast<int>(elementCount),
								   stream);
	CUDA_KERNEL_CHECK();
}

template<class Type, ReduceFunc<Type> F>
__host__ void KCInclusiveScanArray(Type* out, const Type* in, size_t elementCount,
								   cudaStream_t stream = (cudaStream_t)0)
{
	// Delegating to cub here
	size_t bufferSize = 0;
	cub::DeviceScan::InclusiveScan(nullptr, bufferSize,
								   in, out,
								   ReduceWrapper<Type, F>(),
								   static_cast<int>(elementCount),
								   stream);

	DeviceMemory buffer(bufferSize);
	cub::DeviceScan::InclusiveScan(buffer, bufferSize,
								   in, out,
								   ReduceWrapper<Type, F>(),
								   static_cast<int>(elementCount),
								   stream);
	CUDA_KERNEL_CHECK();
}

// Meta Definitions
#define DEFINE_EXCLUSIVE_SCAN(type, func) \
	template \
	__host__ void KCExclusiveScanArray<type, func>(type*, const type*, \
												   size_t, type, \
												   cudaStream_t);

#define DEFINE_INCLUSIVE_SCAN(type, func) \
	template \
	__host__ void KCInclusiveScanArray<type, func>(type*, const type*, \
												   size_t, \
												   cudaStream_t);

#define DEFINE_SCAN_BOTH(type, func) \
	DEFINE_EXCLUSIVE_SCAN(type, func) \
	DEFINE_INCLUSIVE_SCAN(type, func)

#define DEFINE_SCAN_ALL(type) \
	DEFINE_SCAN_BOTH(type, ReduceAdd) \
	DEFINE_SCAN_BOTH(type, ReduceSubtract) \
	DEFINE_SCAN_BOTH(type, ReduceMultiply) \
	DEFINE_SCAN_BOTH(type, ReduceDivide) \
	DEFINE_SCAN_BOTH(type, ReduceMin) \
	DEFINE_SCAN_BOTH(type, ReduceMax)

// Extern Definitions
#define EXTERN_SCAN_BOTH(type, func) \
	extern DEFINE_EXCLUSIVE_SCAN(type, func) \
	extern DEFINE_INCLUSIVE_SCAN(type, func)

#define EXTERN_SCAN_ALL(type) \
	EXTERN_SCAN_BOTH(type, ReduceAdd) \
	EXTERN_SCAN_BOTH(type, ReduceSubtract) \
	EXTERN_SCAN_BOTH(type, ReduceMultiply) \
	EXTERN_SCAN_BOTH(type, ReduceDivide) \
	EXTERN_SCAN_BOTH(type, ReduceMin) \
	EXTERN_SCAN_BOTH(type, ReduceMax)

// Integral Types
EXTERN_SCAN_ALL(int)
EXTERN_SCAN_ALL(unsigned int)
EXTERN_SCAN_ALL(float)
EXTERN_SCAN_ALL(double)
EXTERN_SCAN_ALL(int64_t)
EXTERN_SCAN_ALL(uint64_t)

// Vector Types
EXTERN_SCAN_ALL(Vector2f)
EXTERN_SCAN_ALL(Vector2d)
EXTERN_SCAN_ALL(Vector2i)
EXTERN_SCAN_ALL(Vector2ui)

EXTERN_SCAN_ALL(Vector3f)
EXTERN_SCAN_ALL(Vector3d)
EXTERN_SCAN_ALL(Vector3i)
EXTERN_SCAN_ALL(Vector3ui)

EXTERN_SCAN_ALL(Vector4f)
EXTERN_SCAN_ALL(Vector4d)
EXTERN_SCAN_ALL(Vector4i)
EXTERN_SCAN_ALL(Vector4ui)

// Matrix Types
EXTERN_SCAN_ALL(Matrix2x2f)
EXTERN_SCAN_ALL(Matrix2x2d)
EXTERN_SCAN_ALL(Matrix2x2i)
EXTERN_SCAN_ALL(Matrix2x2ui)

EXTERN_SCAN_ALL(Matrix3x3f)
EXTERN_SCAN_ALL(Matrix3x3d)
EXTERN_SCAN_ALL(Matrix3x3i)
EXTERN_SCAN_ALL(Matrix3x3ui)

EXTERN_SCAN_ALL(Matrix4x4f)
EXTERN_SCAN_ALL(Matrix4x4d)
EXTERN_SCAN_ALL(Matrix4x4i)
EXTERN_SCAN_ALL(Matrix4x4ui)

// Quaternion Types
EXTERN_SCAN_BOTH(QuatF, ReduceMultiply)
EXTERN_SCAN_BOTH(QuatD, ReduceMultiply)