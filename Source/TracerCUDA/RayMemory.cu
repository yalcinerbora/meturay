#include "RayMemory.h"
#include "RayLib/Log.h"
#include "RayLib/MemoryAlignment.h"

#include "CudaSystem.h"
#include "CudaSystem.hpp"
//#include "TracerDebug.h"

#include <type_traits>

#pragma warning( push )
#pragma warning( disable : 4834)
#include <cub/cub.cuh>
#pragma warning( pop )

static constexpr uint32_t INVALID_LOCATION = std::numeric_limits<uint32_t>::max();

struct ValidSplit
{
    __device__ __host__
    __forceinline__ bool operator()(const uint32_t &ids) const
    {
        return (ids != INVALID_LOCATION);
    }
};

__global__ CUDA_LAUNCH_BOUNDS_1D
void FillMatIdsForSortKC(HitKey* gKeys, RayId* gIds,
                         const HitKey* gWorkKeys,
                         uint32_t rayCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount;
        globalId += blockDim.x * gridDim.x)
    {
        gKeys[globalId] = gWorkKeys[globalId];
        gIds[globalId] = globalId;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void ResetHitKeysKC(HitKey* gKeys,
                    HitKey key, uint32_t rayCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount;
        globalId += blockDim.x * gridDim.x)
    {
        gKeys[globalId] = key;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void ResetHitIdsKC(HitKey* gAcceleratorKeys, RayId* gIds,
                   TransformId* gTransformIds,
                   PrimitiveId* gPrimitiveIds,
                   uint32_t identityTransformIndex,
                   uint32_t rayCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount;
        globalId += blockDim.x * gridDim.x)
    {
        gIds[globalId] = globalId;
        gTransformIds[globalId] = identityTransformIndex;
        gPrimitiveIds[globalId] = INVALID_PRIMITIVE_ID;
        gAcceleratorKeys[globalId] = HitKey::InvalidKey;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void FindSplitsSparseKC(uint32_t* gPartLoc,
                        const HitKey* gKeys,
                        const uint32_t locCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < locCount;
        globalId += blockDim.x * gridDim.x)
    {
        HitKey key = gKeys[globalId];
        HitKey keyN = gKeys[globalId + 1];

        uint16_t keyBatch = HitKey::FetchBatchPortion(key);
        uint16_t keyNBatch = HitKey::FetchBatchPortion(keyN);

        // Write location if split is found
        if(keyBatch != keyNBatch) gPartLoc[globalId + 1] = globalId + 1;
        else gPartLoc[globalId + 1] = INVALID_LOCATION;
    }

    // Init first location also
    if((blockIdx.x * blockDim.x + threadIdx.x) == 0)
        gPartLoc[0] = 0;
}

__global__ CUDA_LAUNCH_BOUNDS_1D
void FindSplitBatchesKC(uint16_t* gBatches,
                        const uint32_t* gDenseIds,
                        const HitKey* gSparseKeys,
                        const uint32_t locCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < locCount;
        globalId += blockDim.x * gridDim.x)
    {
        uint32_t index = gDenseIds[globalId];
        HitKey key = gSparseKeys[index];
        gBatches[globalId] = HitKey::FetchBatchPortion(key);
    }
}

void RayMemory::ResizeRayOut(uint32_t rayCount, HitKey baseBoundMatKey)
{
    // Allocate Rays & Work Keys
    GPUMemFuncs::AllocateMultiData(std::tie(dRayOut, dWorkKeys),
                                   memOut,
                                   {rayCount, rayCount});
    // Initialize memory
    if(rayCount != 0)
        leaderDevice.GridStrideKC_X(0, 0, rayCount,
                                    ResetHitKeysKC,
                                    dWorkKeys, baseBoundMatKey,
                                    rayCount);
}

RayMemory::RayMemory(const CudaGPU& g)
    : leaderDevice(g)
{}

void RayMemory::SwapRays()
{
    std::swap(memIn, memOut);
    std::swap(dRayIn, dRayOut);
}

void RayMemory::ResetHitMemory(TransformId identityTransformIndex,
                               uint32_t rayCount, size_t hitStructSize)
{
    CUDA_CHECK(cudaSetDevice(leaderDevice.DeviceId()));

    // Find out sort auxiliary storage
    cub::DoubleBuffer<HitKey::Type> dbKeys(nullptr, nullptr);
    cub::DoubleBuffer<RayId> dbIds(nullptr, nullptr);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, cubSortMemSize,
                                                         dbKeys, dbIds,
                                                         static_cast<int>(rayCount)));
    // Check if while partitioning  double buffer data is
    // enough for using (Unique and Scan) algos
    uint32_t* in = nullptr;
    uint32_t* out = nullptr;
    uint32_t* count = nullptr;
    CUDA_CHECK(cub::DeviceSelect::If(nullptr, cubIfMemSize,
                                     in, out, count,
                                     static_cast<int>(rayCount),
                                     ValidSplit()));

    // Select algo reads from split locations and writes to back buffer Ids (half is used)
    // uses back buffer ids other half as auxiliary buffer
    // This code tries to increase it accordingly
    // Output Count of If also should be considered (add sizeof uint32_t)
    size_t sizeOfTempMemory = std::max(cubSortMemSize, cubIfMemSize + sizeof(uint32_t));
    sizeOfTempMemory = Memory::AlignSize(sizeOfTempMemory);

    Byte* dHitStructBytePtr = nullptr;
    Byte* dTempBytePtr = nullptr;
    GPUMemFuncs::AllocateMultiData(std::tie(dTransformIds, dPrimitiveIds, dHitStructBytePtr,
                                            dIds0, dKeys0, dIds1, dKeys1, dTempBytePtr),
                                   memHit,
                                   {rayCount, rayCount, rayCount * hitStructSize,
                                   rayCount, rayCount, rayCount, rayCount, sizeOfTempMemory});
    // Set Allocated pointers to proper ptrs
    dHitStructs = HitStructPtr(reinterpret_cast<void*>(dHitStructBytePtr),
                               static_cast<int>(hitStructSize));
    dTempMemory = reinterpret_cast<void*>(dTempBytePtr);
    dCurrentIds = dIds0;
    dCurrentKeys = dKeys0;

    // Make nullptr if no hit struct is needed
    if(hitStructSize == 0)
        dHitStructs = HitStructPtr(nullptr, static_cast<int>(hitStructSize));

    // Initialize memory
    leaderDevice.GridStrideKC_X(0, 0, rayCount,
                                ResetHitIdsKC,
                                dCurrentKeys,
                                dCurrentIds,
                                dTransformIds,
                                dPrimitiveIds,
                                identityTransformIndex,
                                static_cast<uint32_t>(rayCount));
}

void RayMemory::SortKeys(RayId*& ids, HitKey*& keys,
                         uint32_t count,
                         const Vector2i& bitMaxValues)
{
    CUDA_CHECK(cudaSetDevice(leaderDevice.DeviceId()));

    // Sort Call over buffers
    HitKey* keysOther = (dCurrentKeys == dKeys0) ? dKeys1 : dKeys0;
    RayId* idsOther = (dCurrentIds == dIds0) ? dIds1 : dIds0;
    cub::DoubleBuffer<HitKey::Type> dbKeys(reinterpret_cast<HitKey::Type*>(dCurrentKeys),
                                           reinterpret_cast<HitKey::Type*>(keysOther));
    cub::DoubleBuffer<RayId> dbIds(dCurrentIds,
                                   idsOther);

    // First sort internals
    int bitStart = 0;
    int bitEnd = bitMaxValues[1];
    if(bitStart != bitEnd)
    {
        CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(dTempMemory, cubSortMemSize,
                                                             dbKeys, dbIds,
                                                             static_cast<int>(count),
                                                             bitStart, bitEnd,
                                                             (cudaStream_t)0,
                                                             false));
    }

    // Then sort batches
    bitStart = HitKey::IdBits;
    bitEnd = HitKey::IdBits + bitMaxValues[0];
    if(bitStart != bitEnd)
    {
        CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(dTempMemory, cubSortMemSize,
                                                             dbKeys, dbIds,
                                                             static_cast<int>(count),
                                                             bitStart, bitEnd,
                                                             (cudaStream_t)0,
                                                             false));
    }

    ids = dbIds.Current();
    keys = reinterpret_cast<HitKey*>(dbKeys.Current());
    dCurrentIds = ids;
    dCurrentKeys = keys;
}

RayPartitions<uint32_t> RayMemory::Partition(uint32_t rayCount)
{
    CUDA_CHECK(cudaSetDevice(leaderDevice.DeviceId()));

    // Use double buffers for partition auxiliary data
    RayId* dEmptyIds = (dCurrentIds == dIds0) ? dIds1 : dIds0;
    HitKey* dEmptyKeys = (dCurrentKeys == dKeys0) ? dKeys1 : dKeys0;

    // Generate Names that make sense for the operation
    // We have total of three buffers
    // Temp Memory will be used for temp memory
    // (it holds enough space for both sort and select)
    //
    // dSparseSplitIndices (a.k.a. dEmptyKeys)
    // will be used as intermediate buffer
    uint32_t* dSparseSplitIndices = reinterpret_cast<uint32_t*>(dEmptyKeys);
    uint32_t* dDenseSplitIndices = reinterpret_cast<uint32_t*>(dEmptyIds);
    uint32_t* dSelectCount = static_cast<uint32_t*>(dTempMemory);
    void* dSelectTempMemory = dSelectCount + 1;

    // Find Split Locations
    // Read from dKeys -> dEmptyKeys
    uint32_t locCount = rayCount - 1;
    leaderDevice.GridStrideKC_X(0, 0, rayCount,
                                FindSplitsSparseKC,
                                dSparseSplitIndices, dCurrentKeys, locCount);

    // Make Splits Dense
    // From dEmptyKeys -> dEmptyIds
    CUDA_CHECK(cub::DeviceSelect::If(dSelectTempMemory, cubIfMemSize,
                                     dSparseSplitIndices, dDenseSplitIndices, dSelectCount,
                                     static_cast<int>(rayCount),
                                     ValidSplit(),
                                     (cudaStream_t)0,
                                     false));

    // Copy Reduced Count
    uint32_t hSelectCount;
    CUDA_CHECK(cudaMemcpy(&hSelectCount, dSelectCount,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Find The Hit Keys for each split
    // From dEmptyIds, dKeys -> dEmptyKeys
    uint16_t* dBatches = reinterpret_cast<uint16_t*>(dSparseSplitIndices);
    leaderDevice.GridStrideKC_X(0, 0, rayCount,
                                FindSplitBatchesKC,
                                dBatches,
                                dDenseSplitIndices,
                                dCurrentKeys,
                                hSelectCount);

    // We need to get dDenseIndices & dDenseKeys
    // Mem copy to vectors
    std::vector<uint16_t> hDenseKeys(hSelectCount);
    std::vector<uint32_t> hDenseIndices(hSelectCount);
    CUDA_CHECK(cudaMemcpy(hDenseKeys.data(), dBatches,
                          sizeof(uint16_t) * hSelectCount,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hDenseIndices.data(), dDenseSplitIndices,
                          sizeof(uint32_t) * hSelectCount,
                          cudaMemcpyDeviceToHost));

    // Construct The Set
    // Add extra index to end as rayCount for cleaner code
    hDenseIndices.push_back(rayCount);
    RayPartitions<uint32_t> partitions;
    for(uint32_t i = 0; i < hSelectCount; i++)
    {
        uint32_t id = hDenseKeys[i];
        uint32_t offset = hDenseIndices[i];
        size_t count = hDenseIndices[i + 1] - hDenseIndices[i];
        partitions.emplace(ArrayPortion<uint32_t>{id, offset, count});
    }
    // Done!
    return std::move(partitions);
}

void RayMemory::FillMatIdsForSort(uint32_t rayCount)
{
    leaderDevice.GridStrideKC_X(0, 0, rayCount,
                                FillMatIdsForSortKC,
                                dCurrentKeys, dCurrentIds, dWorkKeys,
                                rayCount);
}

size_t RayMemory::TotalMemorySize()
{
    return memIn.Size() + memOut.Size() + memHit.Size();
}