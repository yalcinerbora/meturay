
__device__ __forceinline__
uint32_t Delta(uint32_t nodeIndex,
               const uint64_t* gMortonCodes)
{
    uint64_t left = gMortonCodes[nodeIndex];
    uint64_t right = gMortonCodes[nodeIndex + 1];
    uint64_t diffBits = left ^ right;
    uint32_t delta = __clzll(diffBits);
    return delta;
}

template <class Leaf>
__device__ __forceinline__
void ChooseParentAndUpdate(// Output
                           LBVHNode<Leaf>* gNodes,
                           // I-O
                           Vector2ui* gRanges,
                           //
                           const Vector2ui& currentNodeRange,
                           uint32_t currentNode,
                           uint64_t totalNodeCount,
                           //
                           const uint64_t* gMortonCodes,
                           uint32_t internalNodeStartOffset,
                           bool isLeaf)
{
    // Edge Cases
    bool isLeftEdge = (currentNodeRange[0] == 0);
    bool isRightEdge = (currentNodeRange[1] == (totalNodeCount - 1));

    // Check edge cases here (if edge give min so that other will be selected)
    uint32_t leftDelta = isLeftEdge ? 0 : Delta(currentNodeRange[0] - 1, gMortonCodes);
    uint32_t rightDelta = isRightEdge ? 0 : Delta(currentNodeRange[1], gMortonCodes);
    bool parentIsRight = (rightDelta > leftDelta);

    uint32_t globalCurrentNode = (isLeaf) ? currentNode
                                          : (internalNodeStartOffset + currentNode);

    // Parent is right range
    if(parentIsRight)
    {
        uint32_t parent = currentNodeRange[1];
        uint32_t globalParent = internalNodeStartOffset + parent;
        // gNode Hooks are global (gNode is a single array that holds both leaf and internal nodes)
        gNodes[globalParent].body.left = globalCurrentNode;
        gNodes[globalCurrentNode].parent = globalParent;
        // Punch-through the write (don't let it reside on cache only)
        *(reinterpret_cast<volatile uint32_t*>(gRanges + parent) + 0) = currentNodeRange[0];
        printf("%s[%u] R[%u, %u] : P[%u] Left [%u]\n",
               isLeaf ? "L" : "I", currentNode,
               currentNodeRange[0], currentNodeRange[1],
               parent, currentNodeRange[0]);
    }
    // Parent is left range
    else
    {
        uint32_t parent = currentNodeRange[0] - 1;
        uint32_t globalParent = internalNodeStartOffset + parent;
        // gNode Hooks are global (gNode is a single array that holds both leaf and internal nodes)
        gNodes[globalParent].body.right = globalCurrentNode;
        gNodes[globalCurrentNode].parent = globalParent;
        // Punch-through the write (don't let it reside on cache only)
        *(reinterpret_cast<volatile uint32_t*>(gRanges + parent) + 1) = currentNodeRange[1];
        printf("%s[%u] R[%u, %u] : P[%u] Right [%u]\n",
               isLeaf ? "L" : "I", currentNode,
               currentNodeRange[0], currentNodeRange[1],
               parent, currentNodeRange[1]);
    }
}

template <class Leaf, AABBGenFunc<Leaf> AF>
__global__
void KCGenerateAABBs(AABB3f* gAABBs,
                     const Leaf* gLeafs,
                     uint32_t leafCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < leafCount; globalId += blockDim.x * gridDim.x)
    {
        const Leaf leaf = gLeafs[globalId];

        gAABBs[globalId] = AF(leaf);
    }
}

__global__
static void KCGenMortonCodes(uint64_t* gMortonCodes,
                             const AABB3f* gAABBs,
                             const AABB3f& gExtent,
                             const float mortonDelta,
                             uint32_t leafCount)
{
    static constexpr uint32_t AABB_FLOAT_COUNT = 6;
    __shared__ AABB3f sExtent;

    // Push AABB to the shared mem for little bit of perf
    // Use threads to load a word
    if(threadIdx.x <= AABB_FLOAT_COUNT)
    {
        uint32_t loc = sizeof(float) * threadIdx.x;
        Byte* sharedLoc = reinterpret_cast<Byte*>(&sExtent) + loc;
        const Byte* globalLoc = reinterpret_cast<const Byte*>(&gExtent) + loc;
        // Memcpy is fine here nvcc should optimize this out
        // also this is the non-UB version of pointer cast that i know of
        memcpy(sharedLoc, globalLoc, sizeof(float));
    }
    __syncthreads();


    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < leafCount; globalId += blockDim.x * gridDim.x)
    {
        const Vector3f center = gAABBs[globalId].Centroid();

        Vector3f relativePos = center - sExtent.Min();
        assert(relativePos >= Vector3f(0.0f));

        uint32_t x = static_cast<uint32_t>(floor(relativePos[0] / mortonDelta));
        uint32_t y = static_cast<uint32_t>(floor(relativePos[1] / mortonDelta));
        uint32_t z = static_cast<uint32_t>(floor(relativePos[2] / mortonDelta));
        // 64-bit morton code can only hold 21 bit for each value
        assert(x <= (1 << 21) && y <= (1 << 21) &&
               z <= (1 << 21));
        uint64_t code = MortonCode::Compose<uint64_t>(Vector3ui(x, y, z));
        gMortonCodes[globalId] = code;
    }
}

__global__
static void KCCheckUniqueMortonCode(bool& gDoHaveDuplicate,
                                    const uint64_t* gSortedMortonCodes,
                                    uint32_t leafCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < leafCount; globalId += blockDim.x * gridDim.x)
    {
        if(gSortedMortonCodes[globalId] == gSortedMortonCodes[globalId + 1])
            gDoHaveDuplicate = true;
    }
}

template <class Leaf>
__global__
static void KCConstructLinearBVH(uint32_t& gRootIndex,
                                 LBVHNode<Leaf>* gNodes,
                                 Vector2ui* gRanges,
                                 const Leaf* gLeafData,
                                 const uint32_t* gSortedIndices,
                                 const uint64_t* gSortedMortonCodes,
                                 uint32_t leafCount)
{
    uint32_t internalNodeCount = leafCount - 1;
    LBVHNode<Leaf>* gInternalNodes = gNodes + leafCount;
    LBVHNode<Leaf>* gLeafNodes = gNodes;

    // First do the leaf nodes
    // Copy the data
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < leafCount; globalId += blockDim.x * gridDim.x)
    {
        gLeafNodes[globalId].isLeaf = true;
        uint32_t leafIndex = gSortedIndices[globalId];
        gLeafNodes[globalId].leaf = gLeafData[leafIndex];

        Vector2ui myRange = Vector2ui(globalId);
        ChooseParentAndUpdate(gNodes, gRanges, myRange, globalId,
                              leafCount, gSortedMortonCodes, leafCount, true);
    }
    if((blockIdx.x * blockDim.x + threadIdx.x) == 0)
        printf("-------------------------------\n");
    // Do the non-leaf nodes
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < internalNodeCount; globalId += blockDim.x * gridDim.x)
    {
        // Wait over the range value
        // Try to read the value by checking direct value
        // (Force not read from cache)
        Vector2ui myRange;
        do
        {
            // Since warps are processed in lock-step
            // We need to "spin-lock here" but masked warps
            // does spin as well since inter-warp threads may depend
            // on each other
            myRange[0] = *(reinterpret_cast<volatile uint32_t*>(gRanges + globalId) + 0);
            myRange[1] = *(reinterpret_cast<volatile uint32_t*>(gRanges + globalId) + 1);

            // Skip work if data is not ready
            if(myRange[0] == UINT32_MAX || myRange[1] == UINT32_MAX) continue;

            // Root Work
            if(myRange[0] == 0 && myRange[1] == (leafCount - 1))
            {
                gRootIndex = leafCount + globalId;
                gInternalNodes[globalId].parent = UINT32_MAX;
            }
            // Inter-node work
            else ChooseParentAndUpdate(gNodes, gRanges, myRange,
                                       globalId, leafCount,
                                       gSortedMortonCodes, leafCount, false);
            // Just to make use volatile writes goes through i guess?
            __threadfence();

            uint32_t i = 32 - __clz(__activemask()) - 1;
            if((blockIdx.x * blockDim.x + threadIdx.x) == i)
                printf("-------------------------------\n");
        }
        // Do this until all your data is ready
        while(myRange[0] == UINT32_MAX ||
              myRange[1] == UINT32_MAX);

        gInternalNodes[globalId].isLeaf = false;
        AABB3f negAABB = NegativeAABB3f;
        gInternalNodes[globalId].body.aabbMin = negAABB.Min();
        gInternalNodes[globalId].body.aabbMax = negAABB.Max();
    }
}

template <class Leaf>
__global__
static void BottomUpAABBUnion(LBVHNode<Leaf>* gNodes,
                              const AABB3f* gLeafAABBs,
                              uint32_t* gAtomicFlags,
                              uint32_t* gReadyFlags,
                              uint32_t leafCount)
{
    // Leafs first
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < leafCount; globalId += blockDim.x * gridDim.x)
    {
        const uint32_t parentId = gNodes[globalId].parent;
        const uint32_t parentFlagId = parentId - leafCount;
        const uint32_

        // Acquire
        while(atomicXor(gAtomicFlags + parentFlagId, 1u));
        // Union operation

        AABB3f aabb = AABB3f(gNodes[parentId].body.aabbMin,
                             gNodes[parentId].body.aabbMax);
        aabb.UnionSelf(gLeafAABBs[globalId]);
        //
        gNodes[parentId].body.aabbMin = aabb.Min();
        gNodes[parentId].body.aabbMax = aabb.Max();

        printf("[%u] P[%u] Writing [%f, %f, %f] [%f, %f, %f]\n",
               globalId, parentId,
               aabb.Min()[0], aabb.Min()[1], aabb.Min()[2],
               aabb.Max()[0], aabb.Max()[1], aabb.Max()[2]);

        // Atomic increment parent ready flag once
        // Parent thread will activate when it sees 2.
        atomicInc(gReadyFlags + parentFlagId, UINT32_MAX);

        // Lower the flag
        *(reinterpret_cast<volatile int32_t*>(gAtomicFlags + parentFlagId)) = 0;
        __threadfence();
    }
    if((blockIdx.x * blockDim.x + threadIdx.x) == 0)
        printf("-------------------------------\n");
    LBVHNode<Leaf>* gInterNodes = gNodes + leafCount;

    return;

    // Now intermediate nodes
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < (leafCount - 1); globalId += blockDim.x * gridDim.x)
    {
        const uint32_t parentId = gInterNodes[globalId].parent;
        const uint32_t parentFlagId = parentId - leafCount;
        const uint32_t myFlagId = globalId;

        // Wait your data
        // But don't mask the thread here just like in construction
        // or kernel will hang
        volatile const int32_t& gMyAABBReady = *(reinterpret_cast<volatile int32_t*>(gReadyFlags + myFlagId));

        bool aabbReadyReg = false;
        do
        {
            // Do nothing if my AABB is not ready
            if(!aabbReadyReg && gMyAABBReady != 2) continue;
            // Skip querying global memory once aabb is ready
            aabbReadyReg = true;

            // My AABB is ready now do parent union
            bool parentAvail = (atomicXor(gAtomicFlags + parentFlagId, 1u) == 0u);
            if(!parentAvail) continue;


            // Union operation
            AABB3f parentAABB = AABB3f(gNodes[parentId].body.aabbMin,
                                       gNodes[parentId].body.aabbMax);
            AABB3f myAABB = AABB3f(gInterNodes[globalId].body.aabbMin,
                                   gInterNodes[globalId].body.aabbMax);
            myAABB.UnionSelf(parentAABB);
            gNodes[parentId].body.aabbMin = myAABB.Min();
            gNodes[parentId].body.aabbMax = myAABB.Max();

            printf("[%u] P[%u] Writing [%f, %f, %f] [%f, %f, %f]\n",
                   globalId + leafCount, parentId,
                   myAABB.Min()[0], myAABB.Min()[1], myAABB.Min()[2],
                   myAABB.Max()[0], myAABB.Max()[1], myAABB.Max()[2]);

            // Atomic increment parent ready flag once
            // Parent thread will activate when it sees 2.
            atomicInc(gReadyFlags + parentFlagId, UINT32_MAX);

            // Lower the flag
            *(reinterpret_cast<volatile int32_t*>(gAtomicFlags + parentFlagId)) = 0;
            __threadfence();
            break;
        }
        while(true);
    }
}


template <class Leaf, class DF,
          AABBGenFunc<Leaf> AF>
float LinearBVHCPU<Leaf, DF, AF>::FindMortonDelta(const AABB3f& extent)
{
    // Potentially use the entire bitset of the 64-bit int
    Vector3f size = extent.Span();
    // Using-64bit morton code (3 channel)
    // 21-bit per channel
    constexpr uint32_t MORTON_BIT_PER_DIM = 21;
    constexpr double MORTON_RESOLUTION = 1.0f / static_cast<double>(1 << (MORTON_BIT_PER_DIM));
    // Use double precision here (1 / (2^21) is very close to 32-bit precision)
    double dx = static_cast<double>(size[0]) * MORTON_RESOLUTION;
    double dy = static_cast<double>(size[1]) * MORTON_RESOLUTION;
    double dz = static_cast<double>(size[2]) * MORTON_RESOLUTION;
    // Use the largest of the deltas
    return static_cast<float>(std::max({dx, dy, dz}));
}

template <class Leaf, class DF,
          AABBGenFunc<Leaf> AF>
LinearBVHCPU<Leaf, DF, AF>::LinearBVHCPU()
{}

#include "TracerDebug.h"
template <class T>
std::ostream& operator<<(std::ostream& s, const LBVHNode<T>& n)
{
    constexpr uint32_t UINT32_T_MAX = std::numeric_limits<uint32_t>::max();

    s << "P[";
    if(n.parent == UINT32_T_MAX) s << "-";
    else s << n.parent;
    s << "] ";

    if(n.isLeaf)
    {
        s << "LEAF";
    }
    else
    {
        s << "C[";
        if(n.body.left == UINT32_T_MAX) s << "-";
        else s << n.body.left;
        s << ", ";
        if(n.body.right == UINT32_T_MAX) s << "-";
        else s << n.body.right;
        s << "] ";
        s << "AABB[" << n.body.aabbMin << "] [" << n.body.aabbMax << "]";
    }
    return s;
}

template <class Leaf, class DF,
          AABBGenFunc<Leaf> AF>
TracerError LinearBVHCPU<Leaf, DF, AF>::Construct(const Leaf* dLeafList,
                                                  uint32_t leafCount,
                                                  DF df,
                                                  const CudaSystem& system)
{
    // TempMemory (Leaf AABBs, morton code
    AABB3f* dLeafAABBs;
    AABB3f* dExtent;
    DeviceMemory aabbMem;
    GPUMemFuncs::AllocateMultiData(std::tie(dLeafAABBs, dExtent),
                                   aabbMem,
                                   {leafCount, 1});
    uint64_t* dMortonCodes;
    uint32_t* dIndices;
    DeviceMemory mortonAndIndexMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dMortonCodes, dIndices),
                                   mortonAndIndexMemory,
                                   {leafCount, leafCount});

    // Generate AABBs from the leafs
    const CudaGPU& gpu = system.BestGPU();
    gpu.GridStrideKC_X(0, (cudaStream_t)0, leafCount,
                       //
                       KCGenerateAABBs<Leaf, AF>,
                       //
                       dLeafAABBs,
                       dLeafList,
                       leafCount);

    // Find Extent of the leaf cloud
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    ReduceArrayGPU<AABB3f, ReduceAABB3f>(*dExtent,
                                         dLeafAABBs,
                                         leafCount,
                                         NegativeAABB3f);
    // Generate morton code of the AABBs
    AABB3f hExtent;
    CUDA_CHECK(cudaMemcpy(&hExtent, dExtent, sizeof(AABB3f),
                          cudaMemcpyDeviceToHost));
    float mortonDelta = FindMortonDelta(hExtent);
    gpu.GridStrideKC_X(0, (cudaStream_t)0, leafCount,
                       //
                       KCGenMortonCodes,
                       //
                       dMortonCodes,
                       dLeafAABBs,
                       *dExtent,
                       mortonDelta,
                       leafCount);

    // Now sort the morton codes and AABBs
    // Generate sorted memory
    uint32_t* dSortedIndices;
    uint64_t* dSortedMortonCodes;
    bool* dHaveDupMorton;
    DeviceMemory sortedMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dSortedIndices,
                                            dSortedMortonCodes,
                                            dHaveDupMorton),
                                   sortedMemory,
                                   {leafCount, leafCount, 1});

    // Generate index for sort
    // We already set the device above so no need to set again
    IotaGPU(dIndices, 0u, leafCount);
    RadixSortValKeyGPU(dSortedIndices, dSortedMortonCodes,
                       dIndices, dMortonCodes,
                       leafCount);
    gpu.WaitMainStream();
    // Clear the unsorted memory we don't need it anymore
    mortonAndIndexMemory = DeviceMemory();
    dIndices = nullptr;
    dMortonCodes = nullptr;

    // Before construction,
    // check if there are any same morton keyed leafs
    // and return an error (normally you can handle these
    // however it should be rare in our case since we utilize
    // 64-bit morton code (21-bit per dimension meaning
    // 2Mx2Mx2M grid)
    // TODO: handle same keys by doing a split between same coded leafs
    bool hHaveDupMorton = true;
    gpu.GridStrideKC_X(0, (cudaStream_t)0,
                       leafCount - 1,
                       //
                       KCCheckUniqueMortonCode,
                       //
                       *dHaveDupMorton,
                       dSortedMortonCodes,
                       leafCount - 1);
    CUDA_CHECK(cudaMemcpy(&hHaveDupMorton, dHaveDupMorton, sizeof(bool),
                          cudaMemcpyDeviceToHost));
    if(hHaveDupMorton)
    {
        METU_ERROR_LOG("Duplicate Morton Code found on LinearBVH Generation");
        return TracerError(TracerError::TRACER_INTERNAL_ERROR);
    }

    // Now do the (Apetrei 14) (Karras 12) construction
    // By definition there will be (leafCount - 1) internal nodes
    // in the tree
    Vector2ui* dRanges;
    uint32_t* dRootIndex;
    DeviceMemory rangeMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dRanges, dRootIndex),
                                   rangeMemory,
                                   {leafCount - 1, 1});

    // Finally Allocate the entire node array
    LBVHNode<Leaf>* dNodes;
    GPUMemFuncs::AllocateMultiData(std::tie(dNodes),
                                   memory,
                                   {leafCount + leafCount - 1});

    CUDA_CHECK(cudaMemset(dRanges, 0xFF, sizeof(Vector2ui) * (leafCount - 1)));

    // Single GPU Construction Call
    gpu.GridStrideKC_X(0, (cudaStream_t)0,
                       leafCount,
                       //
                       KCConstructLinearBVH<Leaf>,
                       //
                       *dRootIndex,
                       dNodes,
                       dRanges,
                       dLeafList,
                       dSortedIndices,
                       dSortedMortonCodes,
                       leafCount);

    // Host copy the root flag
    uint32_t hRootIndex;
    CUDA_CHECK(cudaMemcpy(&hRootIndex, dRootIndex, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // Bottom-up AABB Union Call
    uint32_t* dAtomicFlags = reinterpret_cast<uint32_t*>(dRanges);
    uint32_t* dReadyFlags = dAtomicFlags + (leafCount - 1);

    CUDA_CHECK(cudaMemset(dAtomicFlags, 0x0, sizeof(uint32_t) * (leafCount - 1)));
    CUDA_CHECK(cudaMemset(dReadyFlags, 0x0, sizeof(uint32_t) * (leafCount - 1)));

    gpu.GridStrideKC_X(0, (cudaStream_t)0,
                       leafCount,
                       //
                       BottomUpAABBUnion<Leaf>,
                       //
                       dNodes,
                       dLeafAABBs,
                       dAtomicFlags,
                       dReadyFlags,
                       leafCount);

    // DEBUG
    Debug::DumpMemToFile("Flags", dAtomicFlags, 2 * (leafCount - 1));
    Debug::DumpMemToFile("LBVHNodes", dNodes, leafCount + leafCount - 1);

    treeGPU.rootIndex = hRootIndex;
    treeGPU.nodes = dNodes;
    treeGPU.nodeCount = leafCount + leafCount - 1;
    treeGPU.leafCount = leafCount;
    treeGPU.DistanceFunction = df;

    return TracerError::OK;
}