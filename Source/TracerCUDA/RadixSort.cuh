#pragma once
/**
Parallel Radix Sort Algorithims
Can defines a custom type

Uses cub backend.

*/

#pragma warning( push )
#pragma warning( disable : 4834)
#include <cub/cub.cuh>
#pragma warning( pop )

#include "RayLib/CudaCheck.h"
#include "DeviceMemory.h"
#include "CudaConstants.h"

template<class Type, bool isAscending = true>
__host__ void KCRadixSortArray(Type* dataOut, const Type* dataIn,
                               size_t elementCount,
                               int bitStart = 0, int bitEnd = sizeof(Type) * 8,
                               cudaStream_t stream = (cudaStream_t)0)
{
    // Delegating to cub here
    size_t bufferSize = 0;
    if(isAscending)
    {
        cub::DeviceRadixSort::SortKeys(nullptr, bufferSize,
                                       dataIn, dataOut,
                                       static_cast<int>(elementCount),
                                       bitStart, bitEnd,
                                       stream);

        DeviceMemory buffer(bufferSize);
        cub::DeviceRadixSort::SortKeys(buffer, bufferSize,
                                       dataIn, dataOut,
                                       static_cast<int>(elementCount),
                                       bitStart, bitEnd,
                                       stream);
    }
    else
    {
        cub::DeviceRadixSort::SortKeysDescending(nullptr, bufferSize,
                                                 dataIn, dataOut,
                                                 static_cast<int>(elementCount),
                                                 bitStart, bitEnd,
                                                 stream);

        DeviceMemory buffer(bufferSize);
        cub::DeviceRadixSort::SortKeysDescending(buffer, bufferSize,
                                                 dataIn, dataOut,
                                                 static_cast<int>(elementCount),
                                                 bitStart, bitEnd,
                                                 stream);
    }
    CUDA_KERNEL_CHECK();
}

template<class Type, class Key, bool isAscending = true>
__host__ void KCRadixSortArray(Type* dataOut, Key* keyOut,
                               const Type* dataIn, const Key* keyIn,
                               size_t elementCount,
                               int bitStart = 0, int bitEnd = sizeof(Key) * 8,
                               cudaStream_t stream = (cudaStream_t)0)
{
    if(isAscending)
    {
        // Delegating to cub here
        size_t bufferSize = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, bufferSize,
                                        keyIn, keyOut,
                                        dataIn, dataOut,
                                        static_cast<int>(elementCount),
                                        bitStart, bitEnd,
                                        stream);

        DeviceMemory buffer(bufferSize);
        cub::DeviceRadixSort::SortPairs(buffer, bufferSize,
                                        keyIn, keyOut,
                                        dataIn, dataOut,
                                        static_cast<int>(elementCount),
                                        bitStart, bitEnd,
                                        stream);
    }
    else
    {
        // Delegating to cub here
        size_t bufferSize = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, bufferSize,
                                        keyIn, keyOut,
                                        dataIn, dataOut,
                                        static_cast<int>(elementCount),
                                        bitStart, bitEnd,
                                        stream);

        DeviceMemory buffer(bufferSize);
        cub::DeviceRadixSort::SortPairsDescending(buffer, bufferSize,
                                        keyIn, keyOut,
                                        dataIn, dataOut,
                                        static_cast<int>(elementCount),
                                        bitStart, bitEnd,
                                        stream);
    }

    CUDA_KERNEL_CHECK();
}

// Meta Definitions
#define DEFINE_RADIX_VALUE(type, order) \
    template \
    __host__ void KCRadixSortArray<type, order>(type*, const type*, \
                                                size_t, int, int, \
                                                cudaStream_t);

#define DEFINE_RADIX_KEY_VALUE(key, type, order) \
    template \
    __host__ void KCRadixSortArray<type, key, order>(type*, key*, \
                                                     const type*, const key*, \
                                                     size_t, int, int, \
                                                     cudaStream_t);

#define DEFINE_RADIX_VALUE_BOTH(type) \
    DEFINE_RADIX_VALUE(type, true); \
    DEFINE_RADIX_VALUE(type, false);

#define DEFINE_RADIX_KEY_VALUE_BOTH(key, type) \
    DEFINE_RADIX_KEY_VALUE(key, type, true); \
    DEFINE_RADIX_KEY_VALUE(key, type, false);

#define EXTERN_RADIX_VALUE_BOTH(type) \
    extern DEFINE_RADIX_VALUE(type, true); \
    extern DEFINE_RADIX_VALUE(type, false);

#define EXTERN_RADIX_KEY_VALUE_BOTH(key, type) \
    extern DEFINE_RADIX_KEY_VALUE(key, type, true); \
    extern DEFINE_RADIX_KEY_VALUE(key, type, false);

#define EXTERN_RADIX_KEY_VALUE_ALL(type) \
    EXTERN_RADIX_KEY_VALUE_BOTH(int, type) \
    EXTERN_RADIX_KEY_VALUE_BOTH(unsigned int, type) \
    EXTERN_RADIX_KEY_VALUE_BOTH(float, type) \
    EXTERN_RADIX_KEY_VALUE_BOTH(double, type) \
    EXTERN_RADIX_KEY_VALUE_BOTH(int64_t, type) \
    EXTERN_RADIX_KEY_VALUE_BOTH(uint64_t, type)

#define DEFINE_RADIX_KEY_VALUE_ALL(type) \
    DEFINE_RADIX_KEY_VALUE_BOTH(int, type) \
    DEFINE_RADIX_KEY_VALUE_BOTH(unsigned int, type) \
    DEFINE_RADIX_KEY_VALUE_BOTH(float, type) \
    DEFINE_RADIX_KEY_VALUE_BOTH(double, type) \
    DEFINE_RADIX_KEY_VALUE_BOTH(int64_t, type) \
    DEFINE_RADIX_KEY_VALUE_BOTH(uint64_t, type)

// Integral Types
EXTERN_RADIX_VALUE_BOTH(int)
EXTERN_RADIX_VALUE_BOTH(unsigned int)
EXTERN_RADIX_VALUE_BOTH(float)
EXTERN_RADIX_VALUE_BOTH(double)
EXTERN_RADIX_VALUE_BOTH(int64_t)
EXTERN_RADIX_VALUE_BOTH(uint64_t)

// ID type sorting
EXTERN_RADIX_KEY_VALUE_ALL(int)
EXTERN_RADIX_KEY_VALUE_ALL(unsigned int)
EXTERN_RADIX_KEY_VALUE_ALL(float)
EXTERN_RADIX_KEY_VALUE_ALL(double)
EXTERN_RADIX_KEY_VALUE_ALL(int64_t)
EXTERN_RADIX_KEY_VALUE_ALL(uint64_t)