#pragma once

#include <set>
#include <cub/cub.cuh>

#include "CudaSystem.h"
#include "ReduceFunctions.cuh"
#include "DeviceMemory.h"

#include "RayLib/ArrayPortion.h"
#include "RayLib/Types.h"
#include "RayLib/BitManipulation.h"

static const uint32_t INVALID_LOCATION = std::numeric_limits<uint32_t>::max();

struct ValidSplit
{
    __device__ __host__ __forceinline__
    bool operator()(const uint32_t& ids) const
    {
        return (ids != INVALID_LOCATION);
    }
};


template <class T, class Key, class FetchFunctor>
__global__
static void KCFetch(Key* gKeyOut,
                    const T* gTypeIn,
                    uint32_t totalNodeCount,
                    FetchFunctor f)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < totalNodeCount; globalId += blockDim.x * gridDim.x)
    {
        gKeyOut[globalId] = f(gTypeIn[globalId]);
    }
}

__global__
static void KCMarkSplits(uint32_t* gPartLoc,
                         const uint32_t* gKeys,
                         const uint32_t locCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < locCount;
        globalId += blockDim.x * gridDim.x)
    {
        uint32_t key = gKeys[globalId];
        uint32_t keyN = gKeys[globalId + 1];

                // Write location if split is found
        if(key != keyN) gPartLoc[globalId + 1] = globalId + 1;
        else gPartLoc[globalId + 1] = INVALID_LOCATION;
    }
}

template<class Type, class Key, class KeyFetchFunctor>
__host__ void PartitionGPU(std::set<ArrayPortion<Key>>& segmentList,
                           DeviceMemory& sortedIndexBuffer,
                           const Type* dArray, size_t elementCount,
                           KeyFetchFunctor f,
                           cudaStream_t stream = (cudaStream_t)0)
{
    static constexpr unsigned int TPB = StaticThreadPerBlock1D;
    uint32_t totalCount = static_cast<uint32_t>(elementCount);

    // Entire List of paths for multiple rays are coming from a single buffer
    // We need to sort by their DTreeIds.
    // Sort as index-id pairs.
    DeviceMemory sortedIndexBuffer0 = DeviceMemory(totalCount * sizeof(uint32_t));
    DeviceMemory sortedIndexBuffer1 = DeviceMemory(totalCount * sizeof(uint32_t));
    DeviceMemory sortedIdBuffer0 = DeviceMemory(totalCount * sizeof(Key));
    DeviceMemory sortedIdBuffer1 = DeviceMemory(totalCount * sizeof(Key));

    cub::DoubleBuffer<Key> dbKeys(static_cast<Key*>(sortedIdBuffer0),
                                  static_cast<Key*>(sortedIdBuffer1));
    cub::DoubleBuffer<uint32_t> dbIndices(static_cast<uint32_t*>(sortedIndexBuffer0),
                                          static_cast<uint32_t*>(sortedIndexBuffer1));

    // Generate Index List
    IotaGPU(static_cast<uint32_t*>(sortedIndexBuffer0), totalCount, 0);

    //
    unsigned int gridSize = static_cast<unsigned int>((totalCount + TPB - 1) / TPB);
    KCFetch<<<gridSize, TPB, 0, stream>>>
    (
        static_cast<uint32_t*>(sortedIdBuffer0),
        dArray,
        totalCount,
        f
    );
    
    // Do not sort all 32-bits
    int bitStart = 0;
    int bitEnd = (totalCount == 0) ? 0 : (Utility::FindLastSet32(totalCount) + 1);

    // Check temp memory requirements of the kernels and alloc mem
    size_t sortTempBufferSize;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sortTempBufferSize,
                                               dbKeys, dbIndices,
                                               static_cast<int>(totalCount),
                                               bitStart, bitEnd));

    size_t ifTempBufferSize;
    CUDA_CHECK(cub::DeviceSelect::If(nullptr, ifTempBufferSize,
                                     static_cast<uint32_t*>(sortedIndexBuffer0), 
                                     static_cast<uint32_t*>(sortedIndexBuffer0),
                                     static_cast<uint32_t*>(sortedIndexBuffer0),
                                     static_cast<int>(totalCount),
                                     ValidSplit()));
    DeviceMemory tempBuffer(std::max(ifTempBufferSize, sortTempBufferSize));

    // Now do the actual sorting
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(static_cast<void*>(tempBuffer), sortTempBufferSize,
                                               dbKeys, dbIndices,
                                               static_cast<int>(totalCount),
                                               bitStart, bitEnd));

    // Check which buffers are used etc.
    // Get one of the buffers for If Kernel output
    DeviceMemory sortedIds, ifOutput;
    if(sortedIdBuffer0 == dbKeys.Current())
    {
        sortedIds = std::move(sortedIdBuffer0);
        ifOutput = std::move(sortedIdBuffer1);
    }        
    else
    {
        sortedIds = std::move(sortedIdBuffer1);
        ifOutput = std::move(sortedIdBuffer0);
    }        
    DeviceMemory sortedIndices, ifInput;
    if(sortedIndexBuffer0 == dbKeys.Current())
    {
        sortedIndices = std::move(sortedIndexBuffer0);
        ifInput = std::move(sortedIndexBuffer1);
    }        
    else
    {
        sortedIndices = std::move(sortedIndexBuffer1);
        ifInput = std::move(sortedIndexBuffer0);
    }    
    //// First find split locations
    //uint32_t locCount = totalNodeCount - 1;
    //bestGPU.GridStrideKC_X(0, 0, totalNodeCount,
    //                       KCMarkSplits,
    //                       ifInput, sortedIds,
    //                       locCount);

    //CUDA_CHECK(cub::DeviceSelect::If(static_cast<void*>(tempBuffer), ifTempBufferSize,
    //                                 ifOutput, ifInput, ---,
    //                                 static_cast<int>(totalNodeCount),
    //                                 ValidSplit()));

    // We have start location of each partition
    //

    // Sorted All nodes now do partition
}