#pragma once
/**

Utility header for header only cuda vector and cpu vector implementations

*/

#include <cstdio>
#include <cassert>

#ifdef METU_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
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
    typedef int cudaError_t;

    inline static constexpr void GPUAssert(cudaError_t code, const char *file, int line) {}
#endif

#ifdef __CUDA_ARCH__
    #define UNROLL_LOOP #pragma unroll
    #define UNROLL_LOOP_COUNT(count) _Pragma(unroll(count))
#else
    #define UNROLL_LOOP
    #define UNROLL_LOOP_COUNT(count)
#endif

#ifdef METU_DEBUG
    constexpr bool METU_DEBUG_BOOL = true;
    #define CUDA_CHECK(func) {GPUAssert((func), __FILE__, __LINE__);}
    #define CUDA_KERNEL_CHECK() \
                CUDA_CHECK(cudaDeviceSynchronize()); \
                CUDA_CHECK(cudaGetLastError())
#else
    constexpr bool METU_DEBUG_BOOL = false;
    //#define CUDA_CHECK(func) func;
    //#define CUDA_KERNEL_CHECK() 
    #define CUDA_CHECK(func) {GPUAssert((func), __FILE__, __LINE__);}
    #define CUDA_KERNEL_CHECK() \
            CUDA_CHECK(cudaGetLastError())

    // TODO: Check this from time to time..
    // Ok after kernels ineed to put get last error
    // in order to properly synchronize i did not understand this
    // hoping for a driver bug instead of some bug resides in the
    // deep dark parts of the code.
#endif


