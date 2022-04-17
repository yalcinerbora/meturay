#pragma once
/**

General Device memory manager for ray and it's auxiliary data

*/

#include <set>

#include "RayLib/ArrayPortion.h"
#include "RayLib/HitStructs.h"

#include "DeviceMemory.h"
#include "RayStructs.h"
#include "ParallelPartition.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

class CudaGPU;

template<class T>
using RayPartitions = std::set<ArrayPortion<T>>;

template<class T>
using RayPartitionsMulti = std::set<MultiArrayPortion<T>>;

template <class KeyType>
using CustomKeySizeEnable = std::enable_if<sizeof(KeyType) == sizeof(HitKey)>::type;

class RayMemory
{
    private:
        // Leader GPU device that is responsible for
        // partitioning and sorting the ray data
        // (Only useful in multi-GPU systems)
        const CudaGPU&              leaderDevice;

        // Ray Related
        DeviceMemory                memIn;
        DeviceMemory                memOut;
        // In rays are going to enter material kernels
        RayGMem*                    dRayIn;
        // Those kernels will output one or multiple rays
        // Each material has a predefined max ray output
        // Out is allocated accordingly then materials fill it
        RayGMem*                    dRayOut;
        //---------------------------------------------------------
        // Hit Related
        // Entire Hit related memory is allocated in bulk.
        DeviceMemory                memHit;
        // MatKey holds the work batch id and material group local id
        // This is used to sort rays to match material kernels
        HitKey*                     dWorkKeys;
        // Transform of the hit
        // Base accelerator fill this value with a potential hit id
        // Leaf accelerators will transform rays to find hit
        TransformId*                dTransformIds;
        // Primitive Id of the hit
        // Inner accelerators fill this value with a primitive group local id
        // Primitive group id can be determined by work group
        PrimitiveId*                dPrimitiveIds;
        // Custom hit Structure allocation pointer
        // This pointer is capable of holding data for all
        // hit structures currently active
        // (i.e. it holds Vec2 bary coords for triangle primitives,
        // hold position for spheres (maybe spherical coords in order to save space).
        // or other custom value for a custom primitive (spline params maybe i dunno)
        HitStructPtr                dHitStructs;
        // --
        // Double buffer and temporary memory for sorting
        // Key/Index pair (key can either be accelerator or material)
        void*                       dTempMemory;
        RayId*                      dIds0, *dIds1;
        HitKey*                     dKeys0, *dKeys1;
        // Current pointers to the double buffer
        // In hit portion of the code it holds accelerator ids etc.
        HitKey*                     dCurrentKeys;
        RayId*                      dCurrentIds;

        // Cub Requires actual tempMemory size
        // It sometimes crash if you give larger memory size
        // I dunno why
        size_t                      cubIfMemSize;
        size_t                      cubSortMemSize;

    public:
        // Constructors & Destructor
                                RayMemory(const CudaGPU& leaderDevice);
                                RayMemory(const RayMemory&) = delete;
                                RayMemory(RayMemory&&) = default;
        RayMemory&              operator=(const RayMemory&) = delete;
        RayMemory&              operator=(RayMemory&&) = delete;
                                ~RayMemory() = default;

        // Accessors
        // Ray In
        RayGMem*                    Rays();
        const RayGMem*              Rays() const;
        // Ray Out
        RayGMem*                    RaysOut();
        const RayGMem*              RaysOut() const;

        // Hit Related
        HitStructPtr                HitStructs();
        const HitStructPtr          HitStructs() const;
        HitKey*                     WorkKeys();
        const HitKey*               WorkKeys() const;
        TransformId*                TransformIds();
        const TransformId*          TransformIds() const;
        PrimitiveId*                PrimitiveIds();
        const PrimitiveId*          PrimitiveIds() const;
        // For sorting
        HitKey*                     CurrentKeys();
        const HitKey*               CurrentKeys() const;
        RayId*                      CurrentIds();
        const RayId*                CurrentIds() const;

        // Misc
        const CudaGPU&              LeaderDevice() const;

        // Memory Allocation
        void                        ResetHitMemory(TransformId identityTransformIndex,
                                                   uint32_t rayCount, size_t hitStructSize);
        void                        ResizeRayOut(uint32_t rayCount, HitKey baseBoundMatKey);
        void                        SwapRays();

        // Utilities
        size_t                      TotalMemorySize();

        // Common Functions
        // Sorts the hit list for multi-kernel calls
        void                        SortKeys(RayId*& ids, HitKey*& keys,
                                             uint32_t count,
                                             const Vector2i& bitRange);
        // Partitions the segments for multi-kernel calls
        RayPartitions<uint32_t>     Partition(uint32_t rayCount);
        // Initialize HitIds and indices
        void                        FillMatIdsForSort(uint32_t rayCount);
        // Memory Usage
        size_t                      UsedGPUMemory() const;

        // Custom Partition Function
        // Partition the rays wrt. to the custom key type
        // This type can be in another Type (FetchType)
        // a device function must be provided for conversion
        // KeyType must fit on a HitKey array
        // thus, this function SFINAE'd the Key type to be same size
        // has HitKey (which is 32-bit number)
        //
        // Partitions are directly created under the device memory,
        // which holds partition offsets,
        // partitioned ray indices are hold in the ray memories inner type
        // Total number of partitions are returned hPartitionCount variable
        //
        // This function may return false if current ray memory could not have
        // enough temporary memory for "sort", "select" (cub functions)
        // operations. Since sizeof(KeyType) == sizeof(HitKey) this should
        // not happen. (However it is provided as a output if cub somehow
        // changes the underlying algorithm of those functions
        template <class KeyType, class FetchType,
                  class FetchFunction,
                  typename = CustomKeySizeEnable<KeyType>>
        bool                PartitionRaysCustom(// Outputs
                                                uint32_t& hPartitionCount,
                                                // GPU Outputs
                                                DeviceMemory & partitionMemory,
                                                uint32_t*& dPartitionOffsets,
                                                KeyType*& dPartitionKeys,
                                                // Inputs
                                                const FetchType* dFetchData,
                                                FetchFunction FetchFunc,
                                                uint32_t rayCount,
                                                const CudaSystem& system);
};

inline const CudaGPU& RayMemory::LeaderDevice() const
{
    return leaderDevice;
}

inline RayGMem* RayMemory::Rays()
{
    return dRayIn;
}

inline const RayGMem* RayMemory::Rays() const
{
    return dRayIn;
}

inline RayGMem* RayMemory::RaysOut()
{
    return dRayOut;
}

inline const RayGMem* RayMemory::RaysOut() const
{
    return dRayOut;
}

inline HitStructPtr RayMemory::HitStructs()
{
    return dHitStructs;
}

inline const HitStructPtr RayMemory::HitStructs() const
{
    return dHitStructs;
}

inline HitKey* RayMemory::WorkKeys()
{
    return dWorkKeys;
}

inline const HitKey* RayMemory::WorkKeys() const
{
    return dWorkKeys;
}

inline TransformId* RayMemory::TransformIds()
{
    return dTransformIds;
}

inline const TransformId* RayMemory::TransformIds() const
{
    return dTransformIds;
}

inline PrimitiveId* RayMemory::PrimitiveIds()
{
    return dPrimitiveIds;
}

inline const PrimitiveId* RayMemory::PrimitiveIds() const
{
    return dPrimitiveIds;
}
//
inline HitKey* RayMemory::CurrentKeys()
{
    return dCurrentKeys;
}
inline const HitKey* RayMemory::CurrentKeys() const
{
    return dCurrentKeys;
}
inline RayId* RayMemory::CurrentIds()
{
    return dCurrentIds;
}

inline const RayId* RayMemory::CurrentIds() const
{
    return dCurrentIds;
}

inline size_t RayMemory::UsedGPUMemory() const
{
    return (memIn.Size() +
            memOut.Size() +
            memHit.Size());
}

template <class KeyType, class FetchType, class FetchFunction>
__global__ static
void KCFillKeysForSort(KeyType* gKeys,
                       uint32_t* gRayIds,
                       const FetchType* gFetchData,
                       FetchFunction FetchFunc,
                       uint32_t rayCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount;
        globalId += blockDim.x * gridDim.x)
    {
        gKeys[globalId] = FetchFunc(gFetchData[globalId]);
        gRayIds[globalId] = globalId;
    }
}

template <class KeyType, class FetchType,
          class FetchFunction, typename>
bool RayMemory::PartitionRaysCustom(// Outputs
                                    uint32_t& hPartitionCount,
                                    // GPU Outputs
                                    DeviceMemory& partitionMemory,
                                    uint32_t*& dPartitionOffsets,
                                    KeyType*& dPartitionKeys,
                                    // Inputs
                                    const FetchType* dFetchData,
                                    FetchFunction FetchFunc,
                                    uint32_t rayCount,
                                    const CudaSystem& system)
{
    hPartitionCount = 0;
    dPartitionOffsets = nullptr;
    dPartitionKeys = nullptr;

    // Type cast the HitKey -> KeyType
    KeyType* dCurKeysT = reinterpret_cast<KeyType*>(dCurrentKeys);
    KeyType* dOtherKeysT = reinterpret_cast<KeyType*>((dCurrentKeys == dKeys0)
                                                      ? dKeys1
                                                      : dKeys0);
    RayId* dIdsOther = (dCurrentIds == dIds0) ? dIds1 : dIds0;

    leaderDevice.GridStrideKC_X(0, 0, rayCount,
                                //
                                KCFillKeysForSort<KeyType, FetchType, FetchFunction>,
                                // Output
                                dCurKeysT,
                                dCurrentIds,
                                // Input
                                dFetchData,
                                FetchFunc,
                                rayCount);

    // Set the leader for cub operations
    CUDA_CHECK(cudaSetDevice(leaderDevice.DeviceId()));

    // Sort the ray ids with keys
    cub::DoubleBuffer<KeyType> dbKeys(dCurKeysT, dOtherKeysT);
    cub::DoubleBuffer<RayId> dbIds(dCurrentIds, dIdsOther);

    // Pre-check if sort operation temp memory size is enough
    size_t requiredSize = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, requiredSize,
                                                         dbKeys, dbIds,
                                                         static_cast<int>(rayCount)));
    if(requiredSize > cubSortMemSize) return false;

    // Looks fine just call the sort
    // since we do not know if 32-bit is fully utilized or not
    // call the radix sort full range [0,32)
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(dTempMemory, cubSortMemSize,
                                                         dbKeys, dbIds,
                                                         static_cast<int>(rayCount)));

    // Get the actual buffer that holds the sorted data
    KeyType* dCurrentKeysT = dbKeys.Current();
    dCurrentIds = dbIds.Current();
    dCurrentKeys = reinterpret_cast<HitKey*>(dCurrentKeysT);
    // Re-purpose the empty buffers for the partition operation
    RayId* dEmptyIds = dbKeys.Alternate();
    KeyType* dEmptyKeysT = dbIds.Alternate();
    // Generate Names that make sense for the operation
    // We have total of three buffers
    // Temp Memory will be used for temp memory
    // (it holds enough space for both sort and select)
    //
    // dSparseSplitIndices (a.k.a. dEmptyKeys)
    // will be used as intermediate buffer
    uint32_t* dSparseSplitIndices = reinterpret_cast<uint32_t*>(dEmptyKeysT);
    uint32_t* dDenseSplitIndices = reinterpret_cast<uint32_t*>(dEmptyIds);
    uint32_t* dSelectCount = static_cast<uint32_t*>(dTempMemory);
    void* dSelectTempMemory = dSelectCount + 1;

    // Find Split Locations
    // Read from dKeys -> dEmptyKeys
    uint32_t locCount = rayCount - 1;
    leaderDevice.GridStrideKC_X(0, 0, locCount,
                                KCMarkSplits<KeyType>,
                                dSparseSplitIndices, dCurrentKeysT, locCount);

    // Make Splits Dense
    // From dEmptyKeys -> dEmptyIds
    CUDA_CHECK(cub::DeviceSelect::If(nullptr, requiredSize,
                                     dSparseSplitIndices, dDenseSplitIndices, dSelectCount,
                                     static_cast<int>(rayCount),
                                     ValidSplit(),
                                     (cudaStream_t)0,
                                     false));
    if(requiredSize > cubIfMemSize) return false;

    // Actual "Partition" algorithm
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

    // Now we can allocate the output buffer since we got the size
    GPUMemFuncs::AllocateMultiData(std::tie(dPartitionOffsets,
                                            dPartitionKeys),
                                   partitionMemory,
                                   {hSelectCount + 1, hSelectCount});

    // Partition offsets are already generated by partition function
    // Copy these to this memory
    // Except the end offset which is the ray count
    CUDA_CHECK(cudaMemcpy(dPartitionOffsets, dDenseSplitIndices,
                          sizeof(uint32_t) * hSelectCount,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dPartitionOffsets + hSelectCount, &rayCount,
                          sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Copy the key of each partition to the dense array
    leaderDevice.GridStrideKC_X(0, 0, hSelectCount,
                                //
                                KCFindSplitBatches<KeyType>,
                                //
                                dPartitionKeys,
                                dDenseSplitIndices,
                                dCurrentKeysT,
                                hSelectCount);
    // All Done!
    hPartitionCount = hSelectCount;
    return true;
}