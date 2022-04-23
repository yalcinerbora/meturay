#pragma once
/**

Utility header for header only cuda vector and cpu vector implementations

*/

#include <cstdio>
#include <cassert>
#include "TracerError.h"
#include "Log.h"

#ifdef METU_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>

    // TODO: Warp size may change in future
    // Then divide this statement for different CC's
    static constexpr uint32_t WARP_SIZE = 32;

    inline static constexpr void GPUAssert(cudaError_t code, const char* file, int line)
    {
        if(code != cudaSuccess)
        {
            METU_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                           fmt::format(fg(fmt::color::green), std::string("CUDA Failure")),
                           cudaGetErrorString(code), file, line);
            assert(false);
        }
    }

    inline static constexpr void GPUMemoryCheck(cudaError_t code)
    {
        //if(code == cudaErrorMemoryAllocation)
        if(code != cudaSuccess)
        {
            //fprintf(stderr, "Cuda Failure: %s %s %d\n", cudaGetErrorString(code), file, line);
            throw TracerException(TracerError::GPU_OUT_OF_MEMORY, cudaGetErrorString(code));
        }
    }
    #define HYBRID_INLINE inline
    #define CUDA_MEMORY_CHECK(func){GPUMemoryCheck((func));}
#else
    #define __device__
    #define __host__
    #define HYBRID_INLINE inline
    typedef int cudaError_t;

    inline static constexpr void GPUAssert(cudaError_t, const char *, int) {}
#endif

#ifdef __CUDA_ARCH__
    #define UNROLL_LOOP _Pragma("unroll")
    #define UNROLL_LOOP_COUNT(count) _Pragma("unroll")(count)
#else
    #define UNROLL_LOOP
    #define UNROLL_LOOP_COUNT(count)
#endif

#if defined(__CUDA_ARCH__) && defined(METU_DEBUG)
    template<class... Args>
    __device__
    static inline void KERNEL_DEBUG_LOG(const char* const string, Args... args)
    {
        printf(string, args...);
    }

#else
    template<class... Args>
    __device__ static inline void KERNEL_DEBUG_LOG(const char* const, Args...){}
#endif

#ifdef METU_DEBUG
    constexpr bool METU_DEBUG_BOOL = true;
    #define CUDA_CHECK(func) GPUAssert((func), __FILE__, __LINE__)
    #define CUDA_CHECK_ERROR(err) GPUAssert(err, __FILE__, __LINE__)
    #define CUDA_KERNEL_CHECK() \
                CUDA_CHECK(cudaDeviceSynchronize()); \
                CUDA_CHECK(cudaGetLastError())
#else
    constexpr bool METU_DEBUG_BOOL = false;
    #define CUDA_CHECK_ERROR(err)
    #define CUDA_KERNEL_PRINTF()
     #define CUDA_CHECK(func) func
     #define CUDA_KERNEL_CHECK()
    //#define CUDA_CHECK(func) GPUAssert((func), __FILE__, __LINE__)
    //#define CUDA_KERNEL_CHECK() \
    //           CUDA_CHECK(cudaDeviceSynchronize()); \
    //           CUDA_CHECK(cudaGetLastError())
    // TODO: Check this from time to time..
#endif
