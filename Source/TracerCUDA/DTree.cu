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

#include <numeric>
#include <algorithm>

#include "TracerDebug.h"

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
static void KCCopyAndAdjustTrees(// Output
                                 DTreeGPU* gDTreesOut,
                                 // Input
                                 const uint32_t* gOldTreeIndices,
                                 const uint32_t* gDTreeOffsets,
                                 // Constants
                                 const DTreeGPU* gDTrees,
                                 const DTreeNode* gDTreeNodes,
                                 uint32_t treeCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < treeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        uint32_t offset = gDTreeOffsets[globalId];
        //uint32_t nodeCount = gDTreeOffsets[globalId + 1] - offset;

        // Copy the old tree directly
        gDTreesOut[globalId] = gDTrees[gOldTreeIndices[globalId]];
        // ->gRoot points to old tree's nodes adjust it
        gDTreesOut[globalId].gRoot = const_cast<DTreeNode*>(gDTreeNodes + offset);
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
        gDTrees[globalId].gRoot = const_cast<DTreeNode*>(gDTreeNodes + offset);
        gDTrees[globalId].nodeCount = nodeCount;

        gDTrees[globalId].irradiance = (setRootIrrad) ? (DTreeGroup::MinIrradiance * 4.0f) : 0.0f;
        gDTrees[globalId].totalSamples = 0;

        // Init very first node
        constexpr DTreeNode nodeMin =
        {
            UINT32_MAX,
            Vector4ui(UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX),
            Vector4f(DTreeGroup::MinIrradiance, DTreeGroup::MinIrradiance,
                     DTreeGroup::MinIrradiance, DTreeGroup::MinIrradiance),
            Vector4ui(0u, 0u, 0u, 0u)
        };
        constexpr DTreeNode nodeZero =
        {
            UINT32_MAX,
            Vector4ui(UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX),
            Vector4f(0.0f, 0.0f, 0.0f, 0.0f),
            Vector4ui(0u, 0u, 0u, 0u)
        };

        // Initialize the root node as well
        *(gDTrees[globalId].gRoot) = (setRootIrrad) ? nodeMin : nodeZero;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCPurgeNodeValues(// Input - Output
                              DTreeNode* gDTreeNodes,
                              // Input
                              uint32_t totalNodeCount)
{
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < totalNodeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        gDTreeNodes[globalId].irradianceEstimates = Zero4f;
        gDTreeNodes[globalId].sampleCounts = Zero4ui;
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCSetNodePointers(// Output
                              DTreeGPU* gDTrees,
                              // Input
                              const DTreeNode* gDTreeNodes,
                              const uint32_t* gDTreeOffsets,
                              const std::pair<uint32_t, float>* dDTreeBaseValues,
                              bool setTreeBases,
                              uint32_t treeCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < treeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        uint32_t offset = gDTreeOffsets[globalId];
        uint32_t nodeCount = gDTreeOffsets[globalId + 1] - offset;
        gDTrees[globalId].gRoot = const_cast<DTreeNode*>(gDTreeNodes + offset);
        gDTrees[globalId].nodeCount = nodeCount;

        if(setTreeBases)
        {
            gDTrees[globalId].totalSamples = dDTreeBaseValues[globalId].first;
            gDTrees[globalId].irradiance = dDTreeBaseValues[globalId].second;
        }
        else
        {
            gDTrees[globalId].totalSamples = 0;
            gDTrees[globalId].irradiance = 0.0f;
        }
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCCopyNodeCount(DTreeGPU* gDTrees,
                            const uint32_t* gAllocators,
                            uint32_t treeCount)
{
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < treeCount;
        globalId += (blockDim.x * gridDim.x))
    {
        gDTrees[globalId].nodeCount = gAllocators[globalId];
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

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCAccumulateRadianceToLeaf(DTreeGPU* gDTrees,
                                       // Input
                                       const PathGuidingNode* gPathNodes,
                                       uint32_t nodeCount,
                                       uint32_t maxPathNodePerRay)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < nodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        const uint32_t nodeIndex = threadId;
        const uint32_t pathStartIndex = nodeIndex / maxPathNodePerRay * maxPathNodePerRay;

        PathGuidingNode gPathNode = gPathNodes[nodeIndex];
        const uint32_t treeIndex = gPathNode.nearestDTreeIndex;

        // Skip if invalid tree
        if(treeIndex == UINT32_MAX) continue;
        // Skip if this node cannot calculate wi
        if(!gPathNode.HasNext()) continue;

        Vector3f wi = gPathNode.Wi<PathGuidingNode>(gPathNodes, pathStartIndex);
        float luminance = Utility::RGBToLuminance(gPathNode.totalRadiance);
        gDTrees[treeIndex].AddRadianceToLeaf(wi, luminance, true);
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
    const uint32_t extraTreeCount = static_cast<uint32_t>(oldTreeIds.size());
    const uint32_t oldTreeCount = static_cast<uint32_t>(hDTreeNodeOffsets.size()) - 1;
    const uint32_t oldNodeCount = hDTreeNodeOffsets.back();
    uint32_t extraNodeCount = 0;
    uint32_t offset = hDTreeNodeOffsets.back();
    newOffsets.reserve(oldTreeIds.size());
    for(auto oldTreeId : oldTreeIds)
    {
        uint32_t nodeCount = hDTreeNodeOffsets[oldTreeId + 1] - hDTreeNodeOffsets[oldTreeId];
        extraNodeCount += nodeCount;
        offset += nodeCount;
        newOffsets.push_back(offset);
    }
    const uint32_t newNodeCount = (oldNodeCount + extraNodeCount);
    uint32_t newSize =  newNodeCount * sizeof(DTreeNode);
    DeviceMemory newNodeMemory = DeviceMemory(newSize);
    DTreeNode* dNewNodes = static_cast<DTreeNode*>(newNodeMemory);

    hDTreeNodeOffsets.insert(hDTreeNodeOffsets.end(),
                             newOffsets.cbegin(),
                             newOffsets.cend());
    const uint32_t newTreeCount = static_cast<uint32_t>(hDTreeNodeOffsets.size() - 1);

    const auto& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    // First copy the old data
    CUDA_CHECK(cudaMemcpy(dNewNodes, dDTreeNodes,
                          oldNodeCount * sizeof(DTreeNode),
                          cudaMemcpyDeviceToDevice));

    // Copy old node data to the newly created trees
    uint32_t i = oldTreeCount;
    for(uint32_t oldTreeId : oldTreeIds)
    {
        uint32_t oldMemOffset = hDTreeNodeOffsets[oldTreeId];
        uint32_t size = hDTreeNodeOffsets[oldTreeId + 1] - oldMemOffset;

        cudaStream_t stream = gpu.DetermineStream();
        CUDA_CHECK(cudaMemcpyAsync(dNewNodes + hDTreeNodeOffsets[i],
                                   dDTreeNodes + oldMemOffset,
                                   size * sizeof(DTreeNode),
                                   cudaMemcpyDeviceToDevice,
                                   stream));
        i++;
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
    DTreeGPU* dDNewTrees = static_cast<DTreeGPU*>(newTreeMemory);

    //
    std::vector<uint32_t> hCopyTreeIndices(oldTreeCount);
    std::iota(hCopyTreeIndices.begin(), hCopyTreeIndices.end(), 0u);
    hCopyTreeIndices.insert(hCopyTreeIndices.end(), oldTreeIds.cbegin(), oldTreeIds.cend());
    DeviceMemory tempMem(hCopyTreeIndices.size() * sizeof(uint32_t));
    uint32_t* dDTreeCopyIndices = static_cast<uint32_t*>(tempMem);
    CUDA_CHECK(cudaMemcpy(dDTreeCopyIndices, hCopyTreeIndices.data(),
                          hCopyTreeIndices.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    gpu.GridStrideKC_X(0, (cudaStream_t)0, newTreeCount,
                       //
                       KCCopyAndAdjustTrees,
                       // Output
                       dDNewTrees,
                       // Input
                       dDTreeCopyIndices,
                       dDTreeNodeOffsets,
                       // Constants
                       dDTrees,
                       dDTreeNodes,
                       newTreeCount);

    // Change the Memory
    treeMemory = std::move(newTreeMemory);
    dDTrees = static_cast<DTreeGPU*>(treeMemory);

    // Done!
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
        0u,
        (cudaStream_t)0
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
        newTreeCount,
        (cudaStream_t)0
    );

    gpu.GridStrideKC_X(0, (cudaStream_t)0, 1,
                       //
                       KCMemset<uint32_t>,
                       //
                       dDTreeNodeOffsets,
                       0u,
                       1);

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

    // Initialize the allocated nodes
    gpu.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                       //
                       KCMemset<DTreeNode>,
                       // Output
                       dDTreeNodes,
                       // Input
                       DTreeNode
                       {
                            std::numeric_limits<uint32_t>::max(),
                            Vector4ui(std::numeric_limits<uint32_t>::max()),
                            Zero4f,
                            Zero4ui
                       },
                       totalNodeCount);

    hDTreeNodeOffsets.resize(newTreeCount + 1);
    CUDA_CHECK(cudaMemcpy(hDTreeNodeOffsets.data(), dDTreeNodeOffsets,
                          hDTreeNodeOffsets.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
}

void  DTreeGroup::DTreeBuffer::InitializeTrees(const std::vector<std::pair<uint32_t, float>>& hDTreeBases,
                                               const std::vector<std::vector<DTreeNode>>& hDTreeNodes,
                                               bool purgeValues,
                                               const CudaSystem& system)
{
    assert(hDTreeBases.size() == hDTreeNodes.size());
    const auto& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));

    hDTreeNodeOffsets.resize(hDTreeNodes.size() + 1);
    hDTreeNodeOffsets.front() = 0;
    std::transform_inclusive_scan(hDTreeNodes.cbegin(), hDTreeNodes.cend(),
                                  std::next(hDTreeNodeOffsets.begin()),
                                  std::plus<uint32_t>{},
                                  [](const auto& vector) -> uint32_t
                                  {
                                      return static_cast<uint32_t>(vector.size());
                                  },
                                  0u);

    // Realloc offset memory
    offsetMemory = DeviceMemory(hDTreeNodeOffsets.size() * sizeof(uint32_t));
    dDTreeNodeOffsets = static_cast<uint32_t*>(offsetMemory);
    CUDA_CHECK(cudaMemcpy(dDTreeNodeOffsets, hDTreeNodeOffsets.data(),
                          hDTreeNodeOffsets.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Realloc Tree Node Memory
    treeNodeMemory = DeviceMemory(hDTreeNodeOffsets.back() * sizeof(DTreeNode));
    dDTreeNodes = static_cast<DTreeNode*>(treeNodeMemory);

    // Copy nodes to GPU Memory
    uint32_t i = 0;
    for(const auto hNodeVector : hDTreeNodes)
    {
        uint32_t nodeOffset = hDTreeNodeOffsets[i];
        uint32_t nodeCount = hDTreeNodeOffsets[i + 1] - nodeOffset;
        DTreeNode* dTreeNodes = dDTreeNodes + nodeOffset;

        CUDA_CHECK(cudaMemcpyAsync(dTreeNodes, hNodeVector.data(),
                                   nodeCount * sizeof(DTreeNode),
                                   cudaMemcpyHostToDevice));
        i++;
    }
    gpu.WaitMainStream();

    // Realloc Tree Memory
    treeMemory = DeviceMemory(sizeof(DTreeGPU) * hDTreeBases.size());
    dDTrees = static_cast<DTreeGPU*>(treeMemory);

    // Push tree base data to gpu temporarily
    using UintFloatPair = std::pair<uint32_t, float>;
    DeviceMemory dTempMemory = DeviceMemory(sizeof(UintFloatPair) * hDTreeBases.size());
    UintFloatPair* dDTreeBaseVals = static_cast<UintFloatPair*>(dTempMemory);
    CUDA_CHECK(cudaMemcpy(dDTreeBaseVals, hDTreeBases.data(),
                          sizeof(UintFloatPair) * hDTreeBases.size(),
                          cudaMemcpyHostToDevice));

    // Adjust node pointers
    uint32_t treeCount = DTreeCount();
    gpu.GridStrideKC_X(0, (cudaStream_t)0, treeCount,
                       //
                       KCSetNodePointers,
                       // Output
                       dDTrees,
                       // Input
                       dDTreeNodes,
                       dDTreeNodeOffsets,
                       dDTreeBaseVals,
                       // Constants
                       !purgeValues,
                       treeCount);

    if(purgeValues)
    {
        uint32_t totalNodeCount = DTreeTotalNodeCount();
        gpu.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                           //
                           KCPurgeNodeValues,
                           // Output
                           dDTreeNodes,
                           totalNodeCount);
    }
}

void DTreeGroup::DTreeBuffer::GetTreeToCPU(DTreeGPU& tree, std::vector<DTreeNode>& nodes, uint32_t treeIndex) const
{
    CUDA_CHECK(cudaMemcpy(&tree, dDTrees + treeIndex, sizeof(DTreeGPU),
                          cudaMemcpyDeviceToHost));
    nodes.resize(tree.nodeCount);
    const DTreeNode* dDTreeNodes = tree.gRoot;
    CUDA_CHECK(cudaMemcpy(nodes.data(), dDTreeNodes,
                          tree.nodeCount * sizeof(DTreeNode),
                          cudaMemcpyDeviceToHost));
}

void DTreeGroup::DTreeBuffer::DumpTreeAsBinary(std::vector<Byte>& data,
                                               uint32_t& nodeCount,
                                               uint32_t treeIndex) const
{
    // Get Data to CPU
    DTreeGPU treeBase;
    std::vector<DTreeNode> nodes;
    GetTreeToCPU(treeBase, nodes, treeIndex);

    // Directly copy it to the buffer
    data.insert(data.end(),
                reinterpret_cast<const Byte*>(&treeBase.totalSamples),
                reinterpret_cast<const Byte*>(&treeBase.totalSamples) + sizeof(uint32_t));
    data.insert(data.end(),
                reinterpret_cast<const Byte*>(&treeBase.irradiance),
                reinterpret_cast<const Byte*>(&treeBase.irradiance) + sizeof(float));
    data.insert(data.end(),
                reinterpret_cast<const Byte*>(nodes.data()),
                reinterpret_cast<const Byte*>(nodes.data()) +
                (sizeof(DTreeNode) * nodes.size()));
    nodeCount = static_cast<uint32_t>(nodes.size());
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

    //// DEBUG
    //for(uint32_t i = 0; i < TreeCount(); i++)
    //{
    //    DTreeGPU tree;
    //    std::vector<DTreeNode> nodes;
    //    GetWriteTreeToCPU(tree, nodes, i);
    //    Debug::DumpMemToFile("AfterBottomUp-writeNodes",
    //                         nodes.data(), nodes.size(),
    //                         (i != 0));
    //}

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
    gpu.WaitMainStream();

    // Reduce each trees requested child count to find out total node count
    SegmentedReduceArrayGPU<uint32_t, ReduceAdd<uint32_t>>
    (
        dTreeChildCounts,
        dNodeChildCounts,
        writeTrees.DTreeNodeOffsetsGPU(),
        writeTrees.DTreeNodeOffsetsGPU() + 1,
        writeTrees.DTreeCount(),
        0u,
        (cudaStream_t)0
    );

    // Add root node (DTree will atleast have a root node)
    // And above kernel only checks if childs should be generated
    // Root does not have any parent so we need to manually include here
    TransformArrayGPU(dTreeChildCounts, treeCount,
                      IncrementFunctor<uint32_t>(),
                      (cudaStream_t)0);

    // Reserve enough nodes on the other tree for construction
    readTrees.ResetAndReserve(dTreeChildCounts, treeCount, system);

    // Set allocation counters to 1
    gpu.GridStrideKC_X(0, (cudaStream_t)0, treeCount,
                       //
                       KCMemset<uint32_t>,
                       //
                       dTreeNodeAllocationCounters,
                       1u, treeCount);

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
    // Copy the actual allocated node count to "tree.nodeCount"
    gpu.GridStrideKC_X(0, (cudaStream_t)0, treeCount,
                       //
                       KCCopyNodeCount,
                       //
                       readTrees.DTrees(),
                       dTreeNodeAllocationCounters,
                       treeCount);
    // Finally swap the trees
    std::swap(readTrees, writeTrees);
}


void DTreeGroup::AddRadiancesFromPaths(const PathGuidingNode* dPGNodes,
                                       uint32_t totalNodeCount,
                                       uint32_t maxPathNodePerRay,
                                       const CudaSystem& system)
{
    const CudaGPU& bestGPU = system.BestGPU();

    bestGPU.GridStrideKC_X(0, (cudaStream_t)0, totalNodeCount,
                           //
                           KCAccumulateRadianceToLeaf,
                           //
                           writeTrees.DTrees(),
                           // Input
                           dPGNodes,
                           totalNodeCount,
                           maxPathNodePerRay);
    bestGPU.WaitMainStream();
}

void DTreeGroup::InitializeTrees(const std::vector<std::pair<uint32_t, float>>& hDTreeBase,
                                 const std::vector<std::vector<DTreeNode>>& hDTreeNodes,
                                 const CudaSystem&system)
{
    readTrees.InitializeTrees(hDTreeBase, hDTreeNodes, false, system);
    writeTrees.InitializeTrees(hDTreeBase, hDTreeNodes, true, system);
}