#pragma once
/**

Utility header for header only cuda vector and cpu vector implementations

*/

#include <cstdio>
#include <cassert>

#ifdef METU_CUDA
	#include <cuda.h>
	#include <cuda_runtime.h>
	#define METU_UNROLL #pragma unroll

	inline static constexpr void GPUAssert(cudaError_t code, const char *file, int line)
	{
		if(code != cudaSuccess)
		{
			fprintf(stderr, "Cuda Failure: %s %s %d\n", cudaGetErrorString(code), file, line);
			assert(false);
		}
	}
#else
	#define __device__
	#define __host__
	#define METU_UNROLL
	typedef int cudaError_t;

	inline static constexpr void GPUAssert(cudaError_t code, const char *file, int line) {}
#endif

#ifdef __CUDA_ARCH__
	#define UNROLL_LOOP #pragma unroll
#else 
	#define UNROLL_LOOP
#endif

#ifdef METU_DEBUG
	#define CUDA_CHECK(func) {GPUAssert((func), __FILE__, __LINE__);}
	#define CUDA_KERNEL_CHECK() \
				CUDA_CHECK(cudaDeviceSynchronize()); \
				CUDA_CHECK(cudaGetLastError());
#else
	#define CUDA_CHECK(func) func;
	#define CUDA_KERNEL_CHECK()
#endif


