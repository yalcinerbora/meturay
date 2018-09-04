#pragma once
/**
Parallel Scan Algorithms
Can define custom reduce function and custom type

Only does exclusive scan

Moderen Reduction using Kepler warp data transfer
http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

Implementation of
https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
*/

#include <cub/cub.cuh>

#include "DeviceMemory.h"
#include "CudaCheck.h"
#include "CudaConstants.h"
#include "OperatorFunctions.cuh"

// We have to wrap out functions to Functors
// Kernels give invalid program counter
// It may mean wrong (enable_if disabled) function is  propogated?
// Wrapping prevent that
template<class Type, ReduceFunc<Type> F>
struct FunctorWrapper
{
	__host__ __device__ __forceinline__
	Type operator()(const Type &a, const Type &b) const
	{
		return F(a, b);
	}
};

template<class Type, ReduceFunc<Type> F>
__host__ void KCExclusiveScanArray(Type* out, const Type* in, 
								   size_t size, Type identityElement, 
								   cudaStream_t stream = (cudaStream_t)0)
{
	// Delegating to cub here
	size_t bufferSize = 0;
	cub::DeviceScan::ExclusiveScan(nullptr, bufferSize,
								   in, out,
								   FunctorWrapper<Type, F>(), 
								   identityElement,
								   static_cast<int>(size),
								   stream);

	DeviceMemory buffer(bufferSize);
	cub::DeviceScan::ExclusiveScan(buffer, bufferSize,
								   in, out,
								   FunctorWrapper<Type, F>(), 
								   identityElement,
								   static_cast<int>(size),
								   stream);
	CUDA_KERNEL_CHECK();
}

template<class Type, ReduceFunc<Type> F>
__host__ void KCInclusiveScanArray(Type* out, const Type* in, size_t size, 
								   cudaStream_t stream = (cudaStream_t)0)
{
	// Delegating to cub here
	size_t bufferSize = 0;
	cub::DeviceScan::InclusiveScan(nullptr, bufferSize,
								   in, out,
								   FunctorWrapper<Type, F>(),
								   static_cast<int>(size),
								   stream);

	DeviceMemory buffer(bufferSize);
	cub::DeviceScan::InclusiveScan(buffer, bufferSize,
								   in, out,
								   FunctorWrapper<Type, F>(),
								   static_cast<int>(size),
								   stream);
	CUDA_KERNEL_CHECK();
}