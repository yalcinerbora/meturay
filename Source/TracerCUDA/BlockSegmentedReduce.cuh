#pragma once

#include <cub/warp/warp_reduce.cuh>
#include "RayLib/CudaCheck.h"

// Segmented version of the Block-wide reduce
// function. Interface tried to be made similar
// like cub interface

template<class T,
         uint32_t TPB_X,
         uint32_t SEGMENT_SIZE>
class BlockSegmentedReduce
{
    private:
    static constexpr uint32_t WARP_PER_SEGMENT = SEGMENT_SIZE / WARP_SIZE;
    static constexpr uint32_t SEGMENT_COUNT = TPB_X / SEGMENT_SIZE;
    static constexpr uint32_t TOTAL_WARP_COUNT = TPB_X / WARP_SIZE;

    using WarpReduce = cub::WarpReduce<T>;
    using WarpTempMem = typename WarpReduce::TempStorage;

    // Don't bother SFINAE this out do static asserts
    static_assert(SEGMENT_SIZE % WARP_SIZE == 0,
                  "Segment size must be multiple of warp size.");
    static_assert(TPB_X % SEGMENT_SIZE == 0,
                  "Block size should be evenly divisible by the segment size.");
    static_assert(SEGMENT_COUNT <= WARP_SIZE,
                  "Each segment should at most have warp size amount of warps.");

    public:
    struct TempStorage
    {
        union
        {
            T           sWarpReductions[SEGMENT_COUNT][WARP_PER_SEGMENT];
            WarpTempMem sWarpTempMem[TOTAL_WARP_COUNT];
        } shMem;
    };

    private:
    TempStorage&        shMem;
    uint32_t            linearLocalId;
    uint32_t            warpId;
    uint32_t            segmentLocalWarpId;
    uint32_t            segmentId;
    uint32_t            laneId;

    protected:
    public:
    // Constructors & Destructor
    __device__          BlockSegmentedReduce(TempStorage&);

    __device__ T        Sum(T threadData,
                            T identityElement);

    // TODO: implement more
    //template<class ReduceOp>
    //__device__ T        Reduce(T threadData,
    //                           T identityElement,
    //                           ReduceOp op);
};

template<class T,
         uint32_t TPB_X,
         uint32_t SEGMENT_SIZE>
__device__ __forceinline__
BlockSegmentedReduce<T, TPB_X, SEGMENT_SIZE>::BlockSegmentedReduce(TempStorage& smem)
    : shMem(smem)
    , linearLocalId(threadIdx.x % SEGMENT_SIZE)
    , warpId(threadIdx.x / WARP_SIZE)
    , laneId(threadIdx.x % WARP_SIZE)
    , segmentLocalWarpId(warpId % WARP_PER_SEGMENT)
    , segmentId(warpId / WARP_PER_SEGMENT)
{}

template<class T,
         uint32_t TPB_X,
         uint32_t SEGMENT_SIZE>
__device__ inline
T BlockSegmentedReduce<T, TPB_X, SEGMENT_SIZE>::Sum(T threadData,
                                                    T identityElement)
{
    auto& sMem = shMem.shMem;

    // Do initial reduction
    T reducedData = WarpReduce(sMem.sWarpTempMem[warpId]).Sum(threadData);
    __syncthreads();

    // Warp leaders (lane0) will have the reduced data
    // Store it to the localized shared memory
    if(laneId == 0)
        sMem.sWarpReductions[segmentId][segmentLocalWarpId] = reducedData;
    __syncthreads();

    // Now segment leader warps will reduce the data
    T data;
    if(segmentLocalWarpId == 0)
    {
        data = (laneId < WARP_PER_SEGMENT) ? sMem.sWarpReductions[segmentId][laneId]
                                           : identityElement;
    }
    __syncthreads();

    reducedData = identityElement;
    if(segmentLocalWarpId == 0)
    {
        reducedData = WarpReduce(sMem.sWarpTempMem[warpId]).Sum(data);
    }
    __syncthreads();
    return reducedData;
}