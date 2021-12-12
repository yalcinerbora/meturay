#include "ScenePositionTree.cuh"
#include "GPUAcceleratorI.h"
#include "ParallelScan.cuh"

#include <cub/cub.cuh>

__global__ static
void KCMarkReduction(uint32_t* gChain,
                     // Inputs
                     const uint64_t* gMortonCodes,
                     float mortonDelta,
                     float areaTreshold,
                     uint32_t totalCount)
{

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < totalCount; globalId += blockDim.x * gridDim.x)
    {
        // Check difference between morton codes
        // If difference exceeds areaThreshold
        // mark chain end
        uint64_t xorDiff = gMortonCodes[globalId] ^ gMortonCodes[globalId + 1];
        uint32_t difference = (sizeof(uint64_t) * BYTE_BITS) - __clzll(xorDiff);
        uint32_t difference3D = (difference + 2) / 3;

        float approxArea = static_cast<float>(difference3D) * mortonDelta;

        if(approxArea >= areaTreshold)
            gChain[globalId] = 1u;
    }
}



__global__ static
void KCGenerateMortonCodes(uint64_t* gMortonCodes,
                           const Vector3f* gPositions,
                           AABB3f extents,
                           float delta,
                           uint32_t positionCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < positionCount; globalId += blockDim.x * gridDim.x)
    {
        Vector3f position = gPositions[globalId];
        Vector3f relativePos = position - extents.Min();
        assert(relativePos >= Vector3f(0.0f));

        uint32_t x = static_cast<uint32_t>(floor(relativePos[0] / delta));
        uint32_t y = static_cast<uint32_t>(floor(relativePos[1] / delta));
        uint32_t z = static_cast<uint32_t>(floor(relativePos[2] / delta));
        // 64-bit can only hold 21 bit for each value
        assert(x + y + z <= 63);

        uint64_t code = MortonCode::Compose<uint64_t>(Vector3ui(x, y, z));
        gMortonCodes[globalId] = code;
    }
}

uint32_t ScenePositionTree::Subdivide(Vector3f* dPositions, Vector3f* dNormals, float* dAreas,
                                      DeviceMemory& tempMemory,
                                      uint32_t elementCount,
                                      const CudaSystem&)
{

}

float ScenePositionTree::FindMortonDelta(const AABB3f& sceneExtents)
{
    // Potentially use the entire bitset of the 64-bit int
    Vector3f size = sceneExtents.Span();
    // Using-64bit morton code (3 channel)
    // 21-bit per channel
    constexpr uint32_t MORTON_BIT_PER_DIM = 21;
    constexpr double MORTON_RESOLUTION = 1.0f / static_cast<double>(1 << (MORTON_BIT_PER_DIM));

    float dx = static_cast<double>(size[0]) * MORTON_RESOLUTION;
    float dy = static_cast<double>(size[1]) * MORTON_RESOLUTION;
    float dz = static_cast<double>(size[2]) * MORTON_RESOLUTION;
    // Use the largest
    return std::max({dx, dy, dz});
}

ScenePositionTree::ScenePositionTree()
{}

TracerError ScenePositionTree::Construct(const AcceleratorBatchMap& sceneAccelerators,
                                         const AABB3f& sceneExtents,
                                         float normalAngleThreshold, float areaThreshold,
                                         const CudaSystem& cudaSystem)
{
    TracerError err = TracerError::OK;
    const CudaGPU& gpu = cudaSystem.BestGPU();

    // Determine total primitive count of the scene
    size_t totalPrimCount = 0;
    std::vector<size_t> offsets;
    for(const auto& [id, acc] : sceneAccelerators)
    {
        offsets.push_back(totalPrimCount);
        size_t leafCount = acc->LeafCount();
        totalPrimCount += leafCount;
    }

    // Get Area, Center & Normal for each primitive
    DeviceMemory tempMemory;
    Vector3f* dPositions;
    Vector3f* dNormals;
    float* dAreas;
    // Allocate
    GPUMemFuncs::AllocateMultiData(std::tie(dPositions, dNormals, dAreas),
                                   tempMemory,
                                   {totalPrimCount, totalPrimCount, totalPrimCount});

    // Ask accelerators to generate data
    uint32_t i = 0;
    for(const auto& [id, acc] : sceneAccelerators)
    {
        acc->RequestPosNormalArea(dPositions + offsets[i],
                                  dNormals + offsets[i],
                                  dAreas + offsets[i],
                                  cudaSystem);
        i++;
    }

    // Subdivide the primitives (if area is too large)
    uint32_t expandedPrimCount = Subdivide(dNormals,
                                           dPositions,
                                           dAreas,
                                           tempMemory,
                                           totalPrimCount,
                                           cudaSystem);
    // Now pointers hold the expanded data (temp memory also expanded)

    // Determine the morton code delta
    float delta = FindMortonDelta(sceneExtents);

    // Generate morton codes
    uint32_t* dIndices;
    uint64_t* dMortonCodes;
    DeviceMemory mortonAndIndexMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dIndices, dMortonCodes),
                                   mortonAndIndexMemory,
                                   {expandedPrimCount, expandedPrimCount});

    // Generate Indices
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    IotaGPU(dIndices, 0u, expandedPrimCount);
    // Generate Codes
    gpu.GridStrideKC_X(0, (cudaStream_t)0, expandedPrimCount,
                       //
                       KCGenerateMortonCodes,
                       //
                       dMortonCodes,
                       dPositions,
                       sceneExtents,
                       delta,
                       expandedPrimCount);


    // Sort indices using morton codes
    uint32_t* dSortedIndices;
    uint64_t* dSortedMortonCodes;
    DeviceMemory sortedMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dSortedIndices, dSortedMortonCodes),
                                   mortonAndIndexMemory,
                                   {expandedPrimCount, expandedPrimCount});

    // TODO: save time by determining how many potential bits are in use
    // However we did maximize the precision so it should be near max bit count
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    RadixSortValKeyGPU(dSortedIndices, dSortedMortonCodes,
                       dIndices, dMortonCodes,
                       expandedPrimCount);

    // Clear unsorted data (we don't need it anymore)
    mortonAndIndexMemory = DeviceMemory();
    dIndices = nullptr;
    dMortonCodes = nullptr;

    // Heuristically Combine nearby data using normal and area
    // Here do check neighbors up to a point
    // Check your positive (next) neighbor and if it is similar,
    // (meaning your combined are is smaller than the threshold and normals are similar)
    // mark as linked
    uint32_t mergedPosCount = 100;
    DeviceMemory chainMem;
    uint32_t* dChain;
    uint32_t* dScanOut;
    GPUMemFuncs::AllocateMultiData(std::tie(dChain, dScanOut),
                                   chainMem,
                                   {expandedPrimCount, expandedPrimCount + 1});
    CUDA_CHECK(cudaMemset(dChain, 0x00, sizeof(uint32_t) * expandedPrimCount));
    // Mark the endpoints on the chain
    gpu.GridStrideKC_X(0, (cudaStream_t)0,
                       expandedPrimCount,
                       //
                       KCMarkReduction,
                       //
                       dChain,
                       // Inputs
                       dSortedMortonCodes,
                       delta,
                       areaThreshold,
                       expandedPrimCount);
    // Determine reduction count
    ExclusiveScanArrayGPU<uint32_t, ReduceAdd<uint32_t>>(dScanOut,
                                                         dChain,
                                                         expandedPrimCount,
                                                         0u);
    //
    //cub::DeviceRunLengthEncode::Encode(d_temp_storage,
    //                                   temp_storage_bytes,
    //                                   d_in,
    //                                   d_unique_out,
    //                                   d_counts_out,
    //                                   d_num_runs_out,
    //                                   num_items);

    //// Check reduction locations
    //SegmentedReduceArrayGPU<>(
    //    dOffse..
    //    )
    //    )




    // Check marks and create final position array
    // Find the unlinked locations on the array
    // Parallel scan the marks to find new total position count & offsets
    // Parallel Reduce the data in between (average positions & normals)
    // and write to the offset location
    DeviceMemory mergedPosMemory;
    Vector3f* dMergedPositions;
    GPUMemFuncs::AllocateMultiData(std::tie(dMergedPositions),
                                   mergedPosMemory,
                                   {mergedPosCount});


    // Compute an LBVH over the location
    if((err = lBVHPoint.Construct(dMergedPositions, mergedPosCount,
                                  cudaSystem)) != TracerError::OK)
        return err;

    // All Done!
    return TracerError::OK;
}