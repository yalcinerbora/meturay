#pragma once

#include <set>
#include <cub/cub.cuh>

#include "CudaSystem.h"
#include "ReduceFunctions.cuh"
#include "DeviceMemory.h"
#include "ParallelSequence.cuh"
#include "ParallelMemset.cuh"

#include "RayLib/ArrayPortion.h"
#include "RayLib/Types.h"
#include "RayLib/BitManipulation.h"

#include "TracerDebug.h"

static const uint32_t INVALID_LOCATION = std::numeric_limits<uint32_t>::max();

struct ValidSplit
{
    __device__ __host__ inline
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

template<class Key>
__global__
static void KCMarkSplits(uint32_t* gPartLoc,
                         const Key* gKeys,
                         const uint32_t locCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < locCount;
        globalId += blockDim.x * gridDim.x)
    {
        Key key = gKeys[globalId];
        Key keyN = gKeys[globalId + 1];

        // Write they if split is found
        if(key != keyN) gPartLoc[globalId + 1] = globalId + 1;
        else gPartLoc[globalId + 1] = INVALID_LOCATION;
    }

    // Init first location also
    if((blockIdx.x * blockDim.x + threadIdx.x) == 0)
        gPartLoc[0] = 0;
}

template <class Key>
__global__
static void KCFindSplitBatches(Key* gDenseKeys,
                               const uint32_t* gDenseIndices,
                               const Key* gSparseKeys,
                               const uint32_t denseCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < denseCount;
        globalId += blockDim.x * gridDim.x)
    {
        uint32_t index = gDenseIndices[globalId];
        Key key = gSparseKeys[index];
        gDenseKeys[globalId] = key;
    }
}


template<class Type, class Key, class KeyFetchFunctor>
__host__ void PartitionGPU(std::set<ArrayPortion<Key>>& segmentList,
                           DeviceMemory& sortedIndexBuffer,
                           const Type* dArray, size_t elementCount,
                           KeyFetchFunctor f,
                           Key maxKey,
                           cudaStream_t stream = (cudaStream_t)0)
{
    // This is memory problem that i was too lazy to fix
    static_assert(sizeof(Key) == sizeof(uint32_t),
                  "Partition code only works with 32-bit width Keys");

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
    IotaGPU(static_cast<uint32_t*>(sortedIndexBuffer0), 0u, totalCount);
    // Generate Keys
    unsigned int gridSize = static_cast<unsigned int>((totalCount + TPB - 1) / TPB);
    KCFetch<<<gridSize, TPB, 0, stream>>>
    (
        static_cast<uint32_t*>(sortedIdBuffer0),
        dArray,
        totalCount,
        f
    );
    CUDA_KERNEL_CHECK();

    // Do not sort all 32-bits
    int bitStart = 0;
    int bitEnd = (maxKey == 0) ? 0 : (Utility::FindLastSet(maxKey) + 1);

    // Check temp memory requirements of the kernels and alloc mem
    size_t sortTempBufferSize;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sortTempBufferSize,
                                               dbKeys, dbIndices,
                                               static_cast<int>(totalCount),
                                               bitStart, bitEnd, stream));

    size_t ifTempBufferSize;
    CUDA_CHECK(cub::DeviceSelect::If(nullptr, ifTempBufferSize,
                                     static_cast<uint32_t*>(sortedIndexBuffer0),
                                     static_cast<uint32_t*>(sortedIndexBuffer0),
                                     static_cast<uint32_t*>(sortedIndexBuffer0),
                                     static_cast<int>(totalCount),
                                     ValidSplit(), stream));
    DeviceMemory tempBuffer(std::max(ifTempBufferSize, sortTempBufferSize));

    // Now do the actual sorting
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(static_cast<void*>(tempBuffer), sortTempBufferSize,
                                               dbKeys, dbIndices,
                                               static_cast<int>(totalCount),
                                               bitStart, bitEnd, stream));

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
    if(sortedIndexBuffer0 == dbIndices.Current())
    {
        sortedIndices = std::move(sortedIndexBuffer0);
        ifInput = std::move(sortedIndexBuffer1);
    }
    else
    {
        sortedIndices = std::move(sortedIndexBuffer1);
        ifInput = std::move(sortedIndexBuffer0);
    }
    // First find split locations
    uint32_t locCount = totalCount - 1;
    gridSize = static_cast<unsigned int>((totalCount + TPB - 1) / TPB);
    KCMarkSplits<<<gridSize, TPB, 0, stream>>>
    (
        static_cast<uint32_t*>(ifInput),
        static_cast<const Key*>(sortedIds),
        locCount
    );
    CUDA_KERNEL_CHECK();
    // We have start location of each partition densely packed
    uint32_t* dSplitCount = static_cast<uint32_t*>(ifOutput);
    uint32_t* dDenseSplitIndices = static_cast<uint32_t*>(ifOutput) + 1;
    CUDA_CHECK(cub::DeviceSelect::If(static_cast<void*>(tempBuffer), ifTempBufferSize,
                                     static_cast<uint32_t*>(ifInput),
                                     dDenseSplitIndices, dSplitCount,
                                     static_cast<int>(totalCount),
                                     ValidSplit(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Load the dense size (so we can allocate)
    uint32_t hSelectCount;
    CUDA_CHECK(cudaMemcpy(&hSelectCount, dSplitCount,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));

    Key* dDenseKeys = static_cast<Key*>(ifInput);
    gridSize = static_cast<unsigned int>((hSelectCount + TPB - 1) / TPB);
    KCFindSplitBatches<<<gridSize, TPB, 0, stream>>>
    (
        dDenseKeys,
        dDenseSplitIndices,
        static_cast<Key*>(sortedIds), // Aka sparse keys
        hSelectCount
    );
    CUDA_KERNEL_CHECK();

    // Fetch these to host
    std::vector<Key> hDenseKeys(hSelectCount);
    std::vector<uint32_t> hDenseIndices(hSelectCount);
    CUDA_CHECK(cudaMemcpy(hDenseKeys.data(), dDenseKeys,
                          sizeof(Key) * hSelectCount,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hDenseIndices.data(), dDenseSplitIndices,
                          sizeof(uint32_t) * hSelectCount,
                          cudaMemcpyDeviceToHost));

    // Sorted All nodes now do partition
    // Construct The Set
    // Add extra index to end as rayCount for cleaner code
    hDenseIndices.push_back(totalCount);
    std::set<ArrayPortion<Key>> partitions;
    for(uint32_t i = 0; i < hSelectCount; i++)
    {
        Key id = hDenseKeys[i];
        uint32_t offset = hDenseIndices[i];
        size_t count = hDenseIndices[i + 1] - hDenseIndices[i];
        partitions.emplace(ArrayPortion<Key>{id, offset, count});
    }
    // Done!
    segmentList = std::move(partitions);
    sortedIndexBuffer = std::move(sortedIndices);
}