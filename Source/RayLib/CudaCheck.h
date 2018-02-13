#pragma once
/**

Utility header for header only cuda vector and cpu vector implementations

*/

#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>


#ifdef METU_CUDA
	#include <cuda.h>
	#include <cuda_runtime.h>
	#define METU_UNROLL #pragma unroll
#else
	#define __device__
	#define __host__
	#define METU_UNROLL
#endif

#ifdef __CUDA_ARCH__
	#define UNROLL_LOOP #pragma unroll
#else 
	#define UNROLL_LOOP
#endif

#ifdef METU_DEBUG
	#define CUDA_CHECK(func) {GPUAssert((func), __FILE__, __LINE__);}
	#define CUDA_KERNEL_CHECK() \
				CUDA_CHECK(cudaGetLastError()); \
				CUDA_CHECK(cudaDeviceSynchronize());
#else
	#define CUDA_CHECK(func) func;
	#define CUDA_KERNEL_CHECK()
#endif


inline static constexpr void GPUAssert(cudaError_t code, const char *file, int line)
{
	if(code != cudaSuccess)
	{
		fprintf(stderr, "Cuda Failure: %s %s %d\n", cudaGetErrorString(code), file, line);
		assert(false);
	}
}