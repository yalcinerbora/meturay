
#include <cub/warp/warp_scan.cuh>
#include "RayLib/CudaCheck.h"

// Segmented version of the Block-wide scan
// function. Interface tried to be made similar
// like cub interface

template<class T,
    uint32_t TPB_X,
    uint32_t SEGMENT_SIZE>
    class BlockSegmentedScan
{
    private:
    static constexpr uint32_t WARP_PER_SEGMENT = SEGMENT_SIZE / WARP_SIZE;
    static constexpr uint32_t SEGMENT_COUNT = TPB_X / SEGMENT_SIZE;
    static constexpr uint32_t TOTAL_WARP_COUNT = TPB_X / WARP_SIZE;

    using WarpScan = cub::WarpScan<T>;
    using WarpTempMem = typename WarpScan::TempStorage;

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
            T           sWarpScanIntermediate[SEGMENT_COUNT][WARP_PER_SEGMENT];
            T           sSegmentAggregates[SEGMENT_COUNT];
            WarpTempMem sWarpTempMem[TOTAL_WARP_COUNT];
        } shMem;
    };

    private:
    TempStorage&        shMem;
    uint32_t            warpId;
    uint32_t            segmentLocalWarpId;
    uint32_t            segmentId;
    uint32_t            laneId;
    uint32_t            segmentLocalThreadId;

    protected:
    public:
    // Constructors & Destructor
    __device__          BlockSegmentedScan(TempStorage&);

    __device__ void     InclusiveSum(T& scanResult,
                                     T& segmentAggregate,
                                     T threadData,
                                     T identityElement);
    __device__ void     InclusiveSum(T& scanResult,
                                     T threadData,
                                     T identityElement);
    __device__ void     ExclusiveSum(T& scanResult,
                                     T& segmentAggregate,
                                     T threadData,
                                     T identityElement);
    __device__ void     ExclusiveSum(T& scanResult,
                                     T threadData,
                                     T identityElement);

    // TODO: implement more
    //template<class ScanOp>
    //__device__ T        InclusiveSum(T& scanResult,
    //                                 T& segmentAggregate,
    //                                 T threadData,
    //                                 T identityElement
    //                                 ScanOp op);
};

template<class T,
         uint32_t TPB_X,
         uint32_t SEGMENT_SIZE>
__device__ __forceinline__
BlockSegmentedScan<T, TPB_X, SEGMENT_SIZE>::BlockSegmentedScan(TempStorage& smem)
    : shMem(smem)
    , warpId(threadIdx.x / WARP_SIZE)
    , laneId(threadIdx.x % WARP_SIZE)
    , segmentLocalWarpId(warpId % WARP_PER_SEGMENT)
    , segmentId(warpId / WARP_PER_SEGMENT)
    , segmentLocalThreadId(segmentLocalWarpId * WARP_SIZE + laneId)
{}

template<class T,
         uint32_t TPB_X,
         uint32_t SEGMENT_SIZE>
__device__ inline
void BlockSegmentedScan<T, TPB_X, SEGMENT_SIZE>::InclusiveSum(T& scanResult,
                                                              T& segmentAggregate,
                                                              T threadData,
                                                              T identityElement)
{
    auto& sMem = shMem.shMem;
    // Do initial reduction
    T intermediateData;
    WarpScan(sMem.sWarpTempMem[warpId]).InclusiveSum(threadData, intermediateData);
    __syncthreads();

    // Last lane in warps (lane31) will have the local total sum
    // Store it to the localized shared memory
    if(laneId == WARP_SIZE - 1)
        sMem.sWarpScanIntermediate[segmentId][segmentLocalWarpId] = intermediateData;
    __syncthreads();

    // Now segment leader warps will scan the data
    T data;
    if(segmentLocalWarpId == 0)
    {
        data = (laneId < WARP_PER_SEGMENT) ? sMem.sWarpScanIntermediate[segmentId][laneId]
                                           : identityElement;
    }
    __syncthreads();

    T warpOffsets = identityElement;
    if(segmentLocalWarpId == 0)
    {
        WarpScan(sMem.sWarpTempMem[segmentId]).ExclusiveSum(data, warpOffsets);
    }
    __syncthreads();

    // Broadcast the final offsets to each warp
    if(segmentLocalWarpId == 0 &&
       laneId < WARP_PER_SEGMENT)
    {
        sMem.sWarpScanIntermediate[segmentId][laneId] = warpOffsets;
    }
    __syncthreads();
    // Store this value for each warp
    T warpOffset = sMem.sWarpScanIntermediate[segmentId][segmentLocalWarpId];
    scanResult = intermediateData + warpOffset;
    __syncthreads();
    // Finally broadcast the segment aggregates
    if(segmentLocalThreadId == (SEGMENT_SIZE - 1))
    {
        sMem.sSegmentAggregates[segmentId] = scanResult;
    }
    __syncthreads();
    segmentAggregate = sMem.sSegmentAggregates[segmentId];
    // All Done!
}

template<class T,
         uint32_t TPB_X,
         uint32_t SEGMENT_SIZE>
__device__ inline
void BlockSegmentedScan<T, TPB_X, SEGMENT_SIZE>::InclusiveSum(T& scanResult,
                                                              T threadData,
                                                              T identityElement)
{
    T aggregate;
    InclusiveSum(scanResult,
                 aggregate,
                 threadData,
                 identityElement);
}

template<class T,
         uint32_t TPB_X,
         uint32_t SEGMENT_SIZE>
__device__ inline
void BlockSegmentedScan<T, TPB_X, SEGMENT_SIZE>::ExclusiveSum(T& scanResult,
                                                              T& segmentAggregate,
                                                              T threadData,
                                                              T identityElement)
{
    auto& sMem = shMem.shMem;
    // Do initial reduction
    T intermediateData;
    WarpScan(sMem.sWarpTempMem[warpId]).ExclusiveSum(threadData, intermediateData);
    __syncthreads();

    // Last lane in warps (lane31) will have the local total sum
    // Store it to the localized shared memory
    if(laneId == WARP_SIZE - 1)
        sMem.sWarpScanIntermediate[segmentId][segmentLocalWarpId] = intermediateData + threadData;
    __syncthreads();

    // Now segment leader warps will scan the data
    T data;
    if(segmentLocalWarpId == 0)
    {
        data = (laneId < WARP_PER_SEGMENT) ? sMem.sWarpScanIntermediate[segmentId][laneId]
                                           : identityElement;
    }
    __syncthreads();

    T warpOffsets = identityElement;
    if(segmentLocalWarpId == 0)
    {
        WarpScan(sMem.sWarpTempMem[segmentId]).ExclusiveSum(data, warpOffsets);
    }
    __syncthreads();

    // Broadcast the final offsets to each warp
    if(segmentLocalWarpId == 0 &&
       laneId < WARP_PER_SEGMENT)
    {
        sMem.sWarpScanIntermediate[segmentId][laneId] = warpOffsets;
    }
    __syncthreads();
    // Store this value for each warp
    T warpOffset = sMem.sWarpScanIntermediate[segmentId][segmentLocalWarpId];

    scanResult = intermediateData + warpOffset;
    // Finally broadcast the segment aggregates
    if(segmentLocalThreadId == (SEGMENT_SIZE - 1))
    {
        sMem.sSegmentAggregates[segmentId] = scanResult;
    }
    __syncthreads();
    segmentAggregate = sMem.sSegmentAggregates[segmentId];
    // All Done!
}

template<class T,
         uint32_t TPB_X,
         uint32_t SEGMENT_SIZE>
__device__ inline
void BlockSegmentedScan<T, TPB_X, SEGMENT_SIZE>::ExclusiveSum(T& scanResult,
                                                              T threadData,
                                                              T identityElement)
{
    T aggregate;
    ExclusiveSum(scanResult,
                 aggregate,
                 threadData,
                 identityElement);
}