#include "DTree.cuh"
#include "DTreeKC.cuh"
#include "CudaSystem.hpp"
#include "ParallelPartition.cuh"
#include "ParallelReduction.cuh"
#include "ParallelTransform.cuh"
#include "ParallelScan.cuh"
#include "ParallelMemset.cuh"
#include "BinarySearch.cuh"

#include "RayLib/Types.h"
#include "RayLib/MemoryAlignment.h"
#include "RayLib/Log.h"
#include "RayLib/CPUTimer.h"

#include "TracerDebug.h"

#include <numeric>

struct FetchTreeIdFunctor
{
    __device__ __host__ __forceinline__
    uint32_t operator()(const PathGuidingNode& node) const
    {
        return node.nearestDTreeIndex;
    }
};

template <class T>
struct IncrementFunctor
{
    __device__ T operator()(T in) { return in + 1; }
};

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCCalculateParentIrradiance(DTreeGPU* gDTrees,
                                        const uint32_t* gTreeIndices,
                                        const uint32_t* gNodeOffsets,
                                        uint32_t totalNodeCount)
{
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < totalNodeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        uint32_t treeIndex = gTreeIndices[globalId];
        uint32_t localNodeIndex = globalId - gNodeOffsets[globalId];

        CalculateParentIrradiance(*(gDTrees + treeIndex),
                                  localNodeIndex);
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCMarkChildRequest(// Output
                               uint32_t* gRequestedChilds,
                               // Input
                               const DTreeGPU* gDTrees,
                               const uint32_t* gTreeIndices,
                               const uint32_t* gNodeOffsets,
                               float fluxRatio,
                               uint32_t totalNodeCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < totalNodeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        uint32_t treeIndex = gTreeIndices[globalId];
        uint32_t localNodeIndex = globalId - gNodeOffsets[globalId];

        // Tree Local output
        uint32_t* gLocalRequestedChilds = gRequestedChilds + gNodeOffsets[globalId];

        MarkChildRequest(gLocalRequestedChilds,
                         *(gDTrees + treeIndex),
                         fluxRatio, localNodeIndex);
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCReconstructEmptyTrees(// Output
                                    DTreeGPU* gDTree,
                                    uint32_t* gTreeNodeAllocationCounters,
                                    // Input
                                    const DTreeGPU* gSiblingTree,
                                    //
                                    const uint32_t* gTreeIndices,
                                    const uint32_t* gNodeOffsets,
                                    //
                                    float fluxRatio,
                                    uint32_t depthLimit,
                                    uint32_t totalNodeCount)
{
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < totalNodeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        uint32_t treeIndex = gTreeIndices[globalId];
        uint32_t localNodeIndex = globalId - gNodeOffsets[globalId];
        uint32_t& gTreeAllocCounter = gTreeNodeAllocationCounters[treeIndex];

        ReconstructEmptyTree(// Output
                             *(gDTree + treeIndex),
                             gTreeAllocCounter,
                             // Input
                             *(gSiblingTree + treeIndex),
                             fluxRatio,
                             depthLimit,
                             localNodeIndex);
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCAddSampleCounts(DTreeGPU* gDTrees,
                              const ArrayPortion<uint32_t>* gPortions,
                              uint32_t portionCount)
{
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < portionCount;
        globalId += (blockDim.x * gridDim.x))
    {
        const auto& gPortion = gPortions[globalId];
        gDTrees[gPortion.portionId].totalSamples += gPortion.count;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCAdjustTreePointers(// Output
                                 DTreeGPU* gDTrees,
                                 // Input
                                 const DTreeNode* gDTreeNodes,
                                 const uint32_t* gDTreeOffsets,
                                 uint32_t treeCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < treeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        uint32_t offset = gDTreeOffsets[globalId];
        uint32_t nodeCount = gDTreeOffsets[globalId + 1] - offset;
        gDTrees->gRoot = const_cast<DTreeNode*>(gDTreeNodes + offset);
        gDTrees->nodeCount = nodeCount;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCAdjustTreePointersAndReset(// Output
                                         DTreeGPU* gDTrees,
                                         // Input
                                         const DTreeNode* gDTreeNodes,
                                         const uint32_t* gDTreeOffsets,
                                         bool setRootIrrad,
                                         uint32_t treeCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < treeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        uint32_t offset = gDTreeOffsets[globalId];
        uint32_t nodeCount = gDTreeOffsets[globalId + 1] - offset;
        gDTrees->gRoot = const_cast<DTreeNode*>(gDTreeNodes + offset);
        gDTrees->nodeCount = nodeCount;

        gDTrees->irradiance = (setRootIrrad) ? (DTreeGroup::MinIrradiance * 4.0f) : 0.0f;
        gDTrees->totalSamples = 0;

        // Init very first node
        constexpr DTreeNode nodeMin =
        {
            UINT32_MAX,
            Vector4ui(UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX),
            Vector4f(DTreeGroup::MinIrradiance, DTreeGroup::MinIrradiance,
                     DTreeGroup::MinIrradiance, DTreeGroup::MinIrradiance)
        };
        constexpr DTreeNode nodeZero =
        {
            UINT32_MAX,
            Vector4ui(UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX),
            Vector4f(0.0f, 0.0f, 0.0f, 0.0f)
        };

        // Initialize the root node as well
        *(gDTrees->gRoot) = (setRootIrrad) ? nodeMin : nodeZero;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCDebugCheckAllocation(bool* gResults,
                                   const DTreeGPU* gDTrees,
                                   const uint32_t* gAllocators,
                                   uint32_t treeCount)
{
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < treeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        bool result = (gDTrees[globalId].nodeCount == gAllocators[globalId]);
        assert(result);
        gResults[globalId] = result;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCDetermineTreeAndOffset(// Output
                                     uint32_t* gNodeTreeIndices,
                                     uint32_t* gNodeOffsets,
                                     // Input
                                     const uint32_t* gDTreeNodeOffsets,
                                     uint32_t treeCount,
                                     uint32_t totalNodeCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < totalNodeCount;
        globalId += (blockDim.x * gridDim.x))

    {
        float index;
        bool found = GPUFunctions::BinarySearchInBetween(index, globalId,
                                                         gDTreeNodeOffsets,
                                                         treeCount + 1);
        assert(found);

        uint32_t indexInt = static_cast<uint32_t>(index);
        gNodeTreeIndices[globalId] = indexInt;
        gNodeOffsets[globalId] = gDTreeNodeOffsets[indexInt];
    }
}

void DTreeGroup::DTreeBuffer::AllocateDefaultTrees(uint32_t count, bool setRootIrrad,
                                                   const CudaSystem& system)
{
    uint32_t currentTreeCount = static_cast<uint32_t>(hDTreeNodeOffsets.size() - 1);
    uint32_t currentNodeCount = hDTreeNodeOffsets.back();

    DeviceMemory newNodeMemory = DeviceMemory((currentNodeCount + count) * sizeof(DTreeNode));
    DeviceMemory newTreeMemory = DeviceMemory((DTreeCount() + count) * sizeof(DTreeGPU));
    DeviceMemory newOffsetMemory = DeviceMemory((DTreeCount() + count + 1) * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(newNodeMemory, treeNodeMemory,
                          treeNodeMemory.Size(),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(newTreeMemory, treeMemory,
                          treeMemory.Size(),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(newOffsetMemory, offsetMemory,
                          offsetMemory.Size(),
                          cudaMemcpyDeviceToDevice));
    treeNodeMemory = std::move(newNodeMemory);
    treeMemory = std::move(newTreeMemory);
    offsetMemory = std::move(newOffsetMemory);
    dDTreeNodes = static_cast<DTreeNode*>(treeNodeMemory);
    dDTrees = static_cast<DTreeGPU*>(treeMemory);
    dDTreeNodeOffsets = static_cast<uint32_t*>(offsetMemory);

    // Set offsets for newly created data
    // on cpu and gpu
    for(uint32_t i = 0; i < count; i++)
    {
        hDTreeNodeOffsets.push_back(hDTreeNodeOffsets.back() + 1);
    }
    CUDA_CHECK(cudaMemcpy(dDTreeNodeOffsets + currentTreeCount + 1,
                          hDTreeNodeOffsets.data() + currentTreeCount + 1,
                          sizeof(uint32_t) * count,
                          cudaMemcpyHostToDevice));

    // Hook the pointers
    const auto& gpu = system.BestGPU();
    gpu.GridStrideKC_X(0, (cudaStream_t)0, count,
                       //
                       KCAdjustTreePointersAndReset,
                       //
                       dDTrees + currentTreeCount,
                       dDTreeNodes,
                       dDTreeNodeOffsets + currentTreeCount,
                       setRootIrrad, count);
}

void DTreeGroup::DTreeBuffer::AllocateExtra(const std::vector<uint32_t>& oldTreeIds,
                                            const CudaSystem& system)
{
    // Determine new size of the Dtree buffer
    std::vector<uint32_t> newOffsets;
    uint32_t oldCount = hDTreeNodeOffsets.back();
    uint32_t extraCount = 0;
    uint32_t offset = hDTreeNodeOffsets.back();
    newOffsets.reserve(oldTreeIds.size());
    for(auto oldTreeId : oldTreeIds)
    {
        uint32_t nodeCount = hDTreeNodeOffsets[oldTreeId + 1] - hDTreeNodeOffsets[oldTreeId];
        extraCount += nodeCount;
        offset += nodeCount;
        newOffsets.push_back(offset);
    }
    uint32_t newSize =  (oldCount + extraCount) * sizeof(DTreeNode);
    DeviceMemory newNodeMemory = DeviceMemory(newSize);
    DTreeNode* dNewNodes = static_cast<DTreeNode*>(newNodeMemory);

    const uint32_t oldTreeCount = static_cast<uint32_t>(hDTreeNodeOffsets.size());
    hDTreeNodeOffsets.insert(hDTreeNodeOffsets.end(),
                             newOffsets.cbegin(),
                             newOffsets.cend());
    const uint32_t newTreeCount = static_cast<uint32_t>(hDTreeNodeOffsets.size() - 1);

    uint32_t i = oldTreeCount;
    const auto& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    // First copy the old data
    CUDA_CHECK(cudaMemcpy(dNewNodes, dDTreeNodes,
                          hDTreeNodeOffsets.back() * sizeof(DTreeNode),
                          cudaMemcpyDeviceToDevice));
    for(uint32_t oldTreeId : oldTreeIds)
    {
        uint32_t oldMemOffset = hDTreeNodeOffsets[oldTreeId];
        uint32_t size = hDTreeNodeOffsets[oldTreeId + 1] - oldMemOffset;

        cudaStream_t stream = gpu.DetermineStream();
        CUDA_CHECK(cudaMemcpyAsync(dDTreeNodes + oldMemOffset,
                                   dNewNodes + hDTreeNodeOffsets[i],
                                   size * sizeof(DTreeNode),
                                   cudaMemcpyDeviceToDevice,
                                   stream));
    }
    gpu.WaitAllStreams();

    // Change the Memory
    treeNodeMemory = std::move(newNodeMemory);
    dDTreeNodes = dNewNodes;

    // Allocate offsets on GPU aswell
    DeviceMemory::EnlargeBuffer(offsetMemory, hDTreeNodeOffsets.size() * sizeof(uint32_t));
    dDTreeNodeOffsets = static_cast<uint32_t*>(offsetMemory);
    CUDA_CHECK(cudaMemcpy(dDTreeNodeOffsets, hDTreeNodeOffsets.data(),
                          hDTreeNodeOffsets.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Change the Actual trees aswell
    DeviceMemory newTreeMemory = DeviceMemory(newTreeCount * sizeof(DTreeGPU));
    CUDA_CHECK(cudaMemcpy(newTreeMemory, treeMemory,
                          treeMemory.Size(), cudaMemcpyDeviceToDevice));
    treeMemory = newTreeMemory;
    dDTrees = static_cast<DTreeGPU*>(treeMemory);

    gpu.GridStrideKC_X(0, (cudaStream_t)0, newTreeCount,
                       //
                       KCAdjustTreePointers,
                       // Output
                       dDTrees,
                       // Input
                       dDTreeNodes,
                       dDTreeNodeOffsets,
                       newTreeCount);

}

void DTreeGroup::DTreeBuffer::ResetAndReserve(const uint32_t* dNewNodeCounts,
                                              uint32_t newTreeCount,
                                              const CudaSystem& system)
{
    const auto& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));

    uint32_t totalNodeCount;
    ReduceArrayGPU<uint32_t, ReduceAdd<uint32_t>, cudaMemcpyDeviceToHost>
    (
        totalNodeCount,
        dNewNodeCounts,
        newTreeCount,
        0u
    );
    gpu.WaitMainStream();

    DeviceMemory::EnlargeBuffer(treeNodeMemory, totalNodeCount * sizeof(DTreeNode));
    dDTreeNodes = static_cast<DTreeNode*>(treeNodeMemory);

    DeviceMemory::EnlargeBuffer(offsetMemory, (newTreeCount + 1) * sizeof(uint32_t));
    dDTreeNodeOffsets = static_cast<uint32_t*>(offsetMemory);

    DeviceMemory::EnlargeBuffer(treeMemory, newTreeCount * sizeof(DTreeGPU));
    dDTrees = static_cast<DTreeGPU*>(treeMemory);

    // Reset the trees etc.
    InclusiveScanArrayGPU<uint32_t, ReduceAdd<uint32_t>>
    (
        dDTreeNodeOffsets + 1,
        dNewNodeCounts,
        newTreeCount
    );
    CUDA_CHECK(cudaMemset(dDTreeNodeOffsets, 0x00, sizeof(uint32_t)));

    gpu.GridStrideKC_X(0, (cudaStream_t)0, newTreeCount,
                       //
                       KCAdjustTreePointersAndReset,
                       // Output
                       dDTrees,
                       // Input
                       dDTreeNodes,
                       dDTreeNodeOffsets,
                       false,
                       newTreeCount);

    hDTreeNodeOffsets.resize(newTreeCount + 1);
    CUDA_CHECK(cudaMemcpy(hDTreeNodeOffsets.data(), dDTreeNodeOffsets,
                          hDTreeNodeOffsets.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
}

void DTreeGroup::DTreeBuffer::GetTreeToCPU(DTreeGPU& tree, std::vector<DTreeNode>& nodes, uint32_t treeIndex) const
{
    CUDA_CHECK(cudaMemcpy(&tree, dDTrees + treeIndex, sizeof(DTreeGPU),
                          cudaMemcpyDeviceToHost));
    nodes.resize(DTreeNodeCount(treeIndex));
    const DTreeNode* dDTreeNodes = tree.gRoot;
    CUDA_CHECK(cudaMemcpy(nodes.data(), dDTreeNodes,
                          DTreeNodeCount(treeIndex) * sizeof(DTreeNode),
                          cudaMemcpyDeviceToHost));
}

void DTreeGroup::DTreeBuffer::DumpTreeAsBinary(std::vector<Byte>& data, uint32_t treeIndex) const
{
    // Get Data to CPU
    DTreeGPU treeBase;
    std::vector<DTreeNode> nodes;
    GetTreeToCPU(treeBase, nodes, treeIndex);

    // Directly copy it to the buffer
    data.insert(data.end(),
                reinterpret_cast<Byte*>(&treeBase.totalSamples),
                reinterpret_cast<Byte*>(&treeBase.totalSamples) + sizeof(uint32_t));
    data.insert(data.end(),
                reinterpret_cast<Byte*>(&treeBase.irradiance),
                reinterpret_cast<Byte*>(&treeBase.irradiance) + sizeof(float));
    data.insert(data.end(),
                reinterpret_cast<Byte*>(nodes.data()),
                reinterpret_cast<Byte*>(nodes.data()) +
                (sizeof(DTreeNode) * nodes.size()));
}

void DTreeGroup::AllocateDefaultTrees(uint32_t count, const CudaSystem& system)
{
    readTrees.AllocateDefaultTrees(count, true, system);
    writeTrees.AllocateDefaultTrees(count, false, system);
}

void DTreeGroup::AllocateExtra(const std::vector<uint32_t>& oldTreeIds,
                               const CudaSystem& system)
{
    readTrees.AllocateExtra(oldTreeIds, system);
    writeTrees.AllocateExtra(oldTreeIds, system);
}

void DTreeGroup::SwapTrees(float fluxRatio, uint32_t depthLimit,
                           const CudaSystem& system)
{
    assert(writeTrees.DTreeCount() == readTrees.DTreeCount());
    const CudaGPU& gpu = system.BestGPU();

    //DTreeGPU treeGPU;
    //std::vector<DTreeNode> nodes;
    //writeTree.DumpTree(treeGPU, nodes);
    //Debug::DumpMemToFile("WT_PC_N", nodes.data(), nodes.size());
    //Debug::DumpMemToFile("WT_PC", &treeGPU, 1);

    //DTreeGPU treeGPU2;
    //std::vector<DTreeNode> nodes2;
    //readTree.DumpTree(treeGPU, nodes);
    //Debug::DumpMemToFile("RT_PC_N", nodes2.data(), nodes2.size());
    //Debug::DumpMemToFile("RT_PC", &treeGPU2, 1);

    // Currently build tree that has its only leafs
    // are valid. Write values to the all nodes
    // Generate an tree index buffer for single kernel launch
    uint32_t totalNodeCount = writeTrees.DTreeTotalNodeCount();
    uint32_t treeCount = writeTrees.DTreeCount();

    // Determine each node's dTree
    DeviceMemory tempMemory;
    uint32_t* dNodeTreeIndices;             // Tree index of each node
    uint32_t* dNodeOffsets;                 // Each tree start offset of each node
    uint32_t* dNodeChildCounts;             // How many child each node needs
    uint32_t* dTreeChildCounts;             // How many child each tree needs
    uint32_t* dTreeNodeAllocationCounters;  // Tree allocation
    DeviceMemory::AllocateMultiData(std::tie(dTreeChildCounts,
                                             dNodeTreeIndices,
                                             dNodeOffsets,
                                             dNodeChildCounts,
                                             dTreeNodeAllocationCounters),
                                    tempMemory,
                                    {treeCount,
                                    totalNodeCount,
                                    totalNodeCount,
                                    totalNodeCount,
                                    treeCount});

    // Set node offsets etc.
    gpu.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                       //
                       KCDetermineTreeAndOffset,
                       //
                       dNodeTreeIndices,
                       dNodeOffsets,
                       //
                       writeTrees.DTreeNodeOffsetsGPU(),
                       writeTrees.DTreeCount(),
                       totalNodeCount);

    // Bottom up calculate parent irradiance
    gpu.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                       //
                       KCCalculateParentIrradiance,
                       //
                       writeTrees.DTrees(),
                       dNodeTreeIndices,
                       dNodeOffsets,
                       totalNodeCount);

    //writeTree.DumpTree(treeGPU, nodes);
    //Debug::DumpMemToFile("WT_AC_N", nodes.data(), nodes.size());
    //Debug::DumpMemToFile("WT_AC", &treeGPU, 1);

    //Byte* dIrrad = reinterpret_cast<Byte*>(readTree.TreeGPU()) + offsetof(DTreeGPU, irradiance);
    //METU_LOG("TOTAL");
    //Debug::DumpMemToStdout(reinterpret_cast<float*>(dIrrad), 1);
    //METU_LOG("===================================================");
    // We have a valid tree now
    // New tree will be reconsturcted from this tree
    // Ask each node that how many child they will need
    gpu.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                       //
                       KCMarkChildRequest,
                       // Output
                       dNodeChildCounts,
                       // Input
                       writeTrees.DTrees(),
                       dNodeTreeIndices,
                       dNodeOffsets,
                       // Constants
                       fluxRatio,
                       totalNodeCount);

    // Sum all values on the GPU
    for(uint32_t treeIndex = 0; treeIndex < readTrees.DTreeCount(); treeIndex++)
    {
        cudaStream_t stream = gpu.DetermineStream();
        uint32_t size = writeTrees.DTreeNodeCount(treeIndex);
        ReduceArrayGPU<uint32_t, ReduceAdd<uint32_t>, cudaMemcpyDeviceToDevice>
        (
            *(dTreeChildCounts + treeIndex),
            dNodeChildCounts + writeTrees.DTreeNodeOffset(treeIndex),
            size,
            0u,
            stream
        );
    }
    gpu.WaitAllStreams();

    // Add root node (DTree will atleast have a root node)
    // And above kernel only checks if childs should be generated
    // Root does not have any parent so we need to manually include here
    TransformArrayGPU(dTreeChildCounts, treeCount,
                      IncrementFunctor<uint32_t>());

    // Reserve enough nodes on the other tree for construction
    readTrees.ResetAndReserve(dTreeChildCounts, treeCount, system);
    // Set allocation counters to 1
    gpu.GridStrideKC_X(0, (cudaStream_t)0, treeCount,
                       //
                       KCMemset<uint32_t>,
                       //
                       dTreeNodeAllocationCounters,
                       1u, treeCount);
    // TODO: is this faster than a single kernel call?
    //CUDA_CHECK(cudaMemset(dTreeNodeAllocationCounters, 0x00, sizeof(uint32_t) * treeCount));
    //CUDA_CHECK(cudaMemset2D(dTreeNodeAllocationCounters, sizeof(uint32_t), 0x01,
    //                        sizeof(uint8_t), treeCount));
    // Reconstruct a new read tree from the findings
    gpu.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                       //
                       KCReconstructEmptyTrees,
                       // Output
                       readTrees.DTrees(),
                       dTreeNodeAllocationCounters,
                       //
                       writeTrees.DTrees(),
                       // Access Related
                       dNodeTreeIndices,
                       dNodeOffsets,
                       // Constants
                       fluxRatio,
                       depthLimit,
                       totalNodeCount);

    // Check that we allocated all the requested nodes
    if constexpr(METU_DEBUG)
    {
        DeviceMemory debugMem(treeCount * sizeof(bool));
        gpu.GridStrideKC_X(0, (cudaStream_t)0, treeCount,
                           //
                           KCDebugCheckAllocation,
                           //
                           static_cast<bool*>(debugMem),
                           readTrees.DTrees(),
                           dTreeNodeAllocationCounters,
                           treeCount);

        static_assert(sizeof(bool) == sizeof(uint8_t));
        std::vector<uint8_t> results(treeCount);
        CUDA_CHECK(cudaMemcpy(results.data(), debugMem,
                              treeCount * sizeof(bool),
                              cudaMemcpyDeviceToHost));
        uint32_t result = std::accumulate
        (
            results.begin(), results.end(), 0u,
            [](uint32_t init, const uint8_t num) -> uint32_t
            {
                return  init + static_cast<uint32_t>(num);
            }
        );
        assert(result == treeCount);
    }
    //readTree.DumpTree(treeGPU, nodes);
    //Debug::DumpMemToFile("RT_FINAL_N", nodes.data(), nodes.size());
    //Debug::DumpMemToFile("RT_FINAL", &treeGPU, 1);

    // Finally swap the trees
    std::swap(readTrees, writeTrees);

    //writeTree.DumpTree(treeGPU, nodes);
    //Debug::DumpMemToFile("WT_FINAL_N", nodes.data(), nodes.size());
    //Debug::DumpMemToFile("WT_FINAL", &treeGPU, 1);

    gpu.WaitAllStreams();
}


void DTreeGroup::AddRadiancesFromPaths(const PathGuidingNode* dPGNodes,
                                       uint32_t totalNodeCount,
                                       uint32_t maxPathNodePerRay,
                                       const CudaSystem& system)
{
    const CudaGPU& bestGPU = system.BestGPU();

    std::set<ArrayPortion<uint32_t>> partitions;
    DeviceMemory sortedIndices;

    CUDA_CHECK(cudaSetDevice(bestGPU.DeviceId()));
    PartitionGPU(partitions, sortedIndices,
                 dPGNodes, totalNodeCount,
                 FetchTreeIdFunctor(),
                 TreeCount());

    std::vector<ArrayPortion<uint32_t>> portionsAsVector;
    portionsAsVector.reserve(partitions.size());

    const GPUList& gpuList = system.SystemGPUs();
    auto currentGPU = gpuList.cbegin();
    // Call kernels
    for(const auto& partition : partitions)
    {
        uint32_t treeIndex = partition.portionId;
        // Skip if these nodes are invalid
        if(treeIndex == InvalidDTreeIndex) continue;

        uint32_t nodeCount = static_cast<uint32_t>(partition.count);

        currentGPU->AsyncGridStrideKC_X(0, nodeCount,
                                        //
                                        KCAccumulateRadianceToLeaf,
                                        //
                                        writeTrees.DTrees() + treeIndex,
                                        static_cast<const uint32_t*>(sortedIndices),
                                        dPGNodes,
                                        nodeCount,
                                        maxPathNodePerRay);
        portionsAsVector.push_back(partition);
        // Get a next GPU if exausted all gpus
        // rool back to start
        currentGPU++;
        if(currentGPU == gpuList.cend()) currentGPU = gpuList.cbegin();
    }

    // Add new samples as sample count
    DeviceMemory tempPortionMem(portionsAsVector.size() * sizeof(ArrayPortion<uint32_t>));
    CUDA_CHECK(cudaMemcpy(tempPortionMem, portionsAsVector.data(),
                          portionsAsVector.size() * sizeof(ArrayPortion<uint32_t>),
                          cudaMemcpyHostToDevice));
    const CudaGPU& gpu = system.BestGPU();
    gpu.GridStrideKC_X(0, (cudaStream_t)0, portionsAsVector.size(),
                       //
                       KCAddSampleCounts,
                       //
                       writeTrees.DTrees(),
                       static_cast<const ArrayPortion<uint32_t>*>(tempPortionMem),
                       static_cast<uint32_t>(portionsAsVector.size()));

    // Wait all gpus to finish
    system.SyncAllGPUs();
}

//
//void DTree::DTreeBuffer::FixPointers()
//{
//    Byte* nodeStart = static_cast<Byte*>(memory) + AlignedOffsetDTreeGPU;
//    Byte* nodePtrLoc = static_cast<Byte*>(memory) + offsetof(DTreeGPU, gRoot);
//    CUDA_CHECK(cudaMemcpy(nodePtrLoc, &nodeStart, sizeof(DTreeNode*),
//                          cudaMemcpyHostToDevice));
//}
//
//DTree::DTreeBuffer::DTreeBuffer()
//    : dDTree(nullptr)
//    , nodeCount(0)
//{
//    nodeCount = 1;
//    DeviceMemory::EnlargeBuffer(memory, AlignedOffsetDTreeGPU + sizeof(DTreeNode));
//    dDTree = static_cast<DTreeGPU*>(memory);
//    DTreeNode* dDTreeNodes = reinterpret_cast<DTreeNode*>(static_cast<Byte*>(memory) + AlignedOffsetDTreeGPU);
//
//    // Init Tree
//    DTreeGPU hDTree;
//    hDTree.gRoot = dDTreeNodes;
//    hDTree.nodeCount = 1;
//    hDTree.irradiance = 0.0f;
//    hDTree.totalSamples = 0;
//    CUDA_CHECK(cudaMemcpy(dDTree, &hDTree, sizeof(DTreeGPU),
//                          cudaMemcpyHostToDevice));
//
//    // Init very first node
//    DTreeNode hNode;
//    hNode.irradianceEstimates = Zero4;
//    hNode.childIndices = Vector4ui(std::numeric_limits<uint32_t>::max());
//    hNode.parentIndex = std::numeric_limits<uint16_t>::max();
//    CUDA_CHECK(cudaMemcpy(dDTreeNodes, &hNode, sizeof(DTreeNode),
//                          cudaMemcpyHostToDevice));
//}
//
//DTree::DTreeBuffer::DTreeBuffer(const DTreeBuffer& other)
//    : memory(other.memory.Size())
//    , nodeCount(other.nodeCount)
//    , dDTree(static_cast<DTreeGPU*>(memory))
//{
//    CUDA_CHECK(cudaMemcpy(memory, other.memory,
//                          AlignedOffsetDTreeGPU + nodeCount * sizeof(DTreeNode),
//                          cudaMemcpyDeviceToDevice));
//    FixPointers();
//}
//
//DTree::DTreeBuffer& DTree::DTreeBuffer::operator=(const DTreeBuffer& other)
//{
//    DeviceMemory::EnlargeBuffer(memory, other.memory.Size());
//    nodeCount = other.nodeCount;
//    dDTree = static_cast<DTreeGPU*>(memory);
//
//    CUDA_CHECK(cudaMemcpy(memory, other.memory,
//                          AlignedOffsetDTreeGPU + nodeCount * sizeof(DTreeNode),
//                          cudaMemcpyDeviceToDevice));
//    FixPointers();
//    return *this;
//}
//
//void DTree::DTreeBuffer::ResetAndReserve(size_t newNodeCount,
//                                         const CudaGPU& gpu,
//                                         cudaStream_t stream)
//{
//    // Check capacity and if its not large enough
//    // allocate larger memory
//    size_t capacity = (memory.Size() - AlignedOffsetDTreeGPU) / sizeof(DTreeNode);
//    if(capacity < newNodeCount)
//    {
//        size_t size = AlignedOffsetDTreeGPU + (newNodeCount * sizeof(DTreeNode));
//        DeviceMemory::EnlargeBuffer(memory, size);
//        dDTree = static_cast<DTreeGPU*>(memory);
//        FixPointers();
//    }
//    // Reset all node values
//    gpu.GridStrideKC_X(0, stream, newNodeCount,
//                        //
//                       KCInitDTreeNodes,
//                       //
//                       dDTree,
//                       static_cast<uint32_t>(newNodeCount));
//
//    nodeCount = 0;
//}
//
//void DTree::DTreeBuffer::CopyGPUNodeCountToCPU(cudaStream_t stream)
//{
//    CUDA_CHECK(cudaMemcpyAsync(&nodeCount, reinterpret_cast<Byte*>(dDTree) + offsetof(DTreeGPU, nodeCount),
//                               sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
//}
//
//void DTree::DTreeBuffer::DumpTree(DTreeGPU& treeCPU, std::vector<DTreeNode>& nodesCPU) const
//{
//    CUDA_CHECK(cudaMemcpy(&treeCPU, dDTree, sizeof(DTreeGPU),
//                          cudaMemcpyDeviceToHost));
//    nodesCPU.resize(nodeCount);
//    const DTreeNode* dDTreeNodes = treeCPU.gRoot;
//    CUDA_CHECK(cudaMemcpy(nodesCPU.data(), dDTreeNodes, nodeCount * sizeof(DTreeNode),
//                          cudaMemcpyDeviceToHost));
//}
//
//void DTree::DTreeBuffer::DumpTreeAsBinary(std::vector<Byte>& data) const
//{
//    // Get Data to CPU
//    DTreeGPU treeBase;
//    std::vector<DTreeNode> nodes;
//    DumpTree(treeBase, nodes);
//
//    // Directly copy it to the buffer
//    data.insert(data.end(),
//                reinterpret_cast<Byte*>(&treeBase.totalSamples),
//                reinterpret_cast<Byte*>(&treeBase.totalSamples) + sizeof(uint32_t));
//    data.insert(data.end(),
//                reinterpret_cast<Byte*>(&treeBase.irradiance),
//                reinterpret_cast<Byte*>(&treeBase.irradiance) + sizeof(float));
//    data.insert(data.end(),
//                reinterpret_cast<Byte*>(nodes.data()),
//                reinterpret_cast<Byte*>(nodes.data()) +
//                (sizeof(DTreeNode) * nodes.size()));
//}
//

//
//void DTree::AddRadiancesFromPaths(const uint32_t* dNodeIndexArray,
//                                  const PathGuidingNode* dPathNodes,
//                                  const ArrayPortion<uint32_t>& portion,
//                                  uint32_t maxPathNodePerRay,
//                                  const CudaGPU& gpu)
//{
//    cudaStream_t stream = gpu.DetermineStream();
//    uint32_t nodeCount = static_cast<uint32_t>(portion.count);
//
//    gpu.GridStrideKC_X(0, stream, portion.count,
//                       //
//                       KCAccumulateRadianceToLeaf,
//                       //
//                       writeTree.TreeGPU(),
//                       dNodeIndexArray,
//                       dPathNodes,
//                       nodeCount,
//                       maxPathNodePerRay);
//
//    uint32_t totalSampleCount = static_cast<uint32_t>(portion.count);
//    uint32_t hSamples;
//    CUDA_CHECK(cudaMemcpy(&hSamples, &writeTree.TreeGPU()->totalSamples,
//                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
//    hSamples += totalSampleCount;
//    CUDA_CHECK(cudaMemcpy(&writeTree.TreeGPU()->totalSamples, &hSamples,
//                          sizeof(uint32_t), cudaMemcpyHostToDevice));
//}
//
//void DTree::GetReadTreeToCPU(DTreeGPU& tree, std::vector<DTreeNode>& nodes) const
//{
//    readTree.DumpTree(tree, nodes);
//}
//void DTree::GetWriteTreeToCPU(DTreeGPU& tree, std::vector<DTreeNode>& nodes) const
//{
//    writeTree.DumpTree(tree, nodes);
//}
//
//void DTree::DumpTreeAsBinary(std::vector<Byte>& data, bool fetchReadTree) const
//{
//    if(fetchReadTree)
//        readTree.DumpTreeAsBinary(data);
//    else
//        writeTree.DumpTreeAsBinary(data);
//}