
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
        s << n.leaf;
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

__device__ inline
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
__device__ inline
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
    uint32_t rangeIndexInt, value;
    if(parentIsRight)
    {
        uint32_t parent = currentNodeRange[1];
        uint32_t globalParent = internalNodeStartOffset + parent;
        // gNode Hooks are global (gNode is a single array that holds both leaf and internal nodes)
        gNodes[globalParent].body.left = globalCurrentNode;
        gNodes[globalCurrentNode].parent = globalParent;

        //// Punch-through the write (don't let it reside on cache (only)
        //*(reinterpret_cast<volatile uint32_t*>(gRanges + parent) + 0) = currentNodeRange[0];
        rangeIndexInt = 2 * parent + 0;
        value = currentNodeRange[0];
    }
    // Parent is left range
    else
    {
        uint32_t parent = currentNodeRange[0] - 1;
        uint32_t globalParent = internalNodeStartOffset + parent;
        // gNode Hooks are global (gNode is a single array that holds both leaf and internal nodes)
        gNodes[globalParent].body.right = globalCurrentNode;
        gNodes[globalCurrentNode].parent = globalParent;
        //// Punch-through the write (don't let it reside on cache only)
        //*(reinterpret_cast<volatile uint32_t*>(gRanges + parent) + 1) = currentNodeRange[1];
        rangeIndexInt = 2 * parent + 1;
        value = currentNodeRange[1];
    }

    uint32_t* gRangesInt = reinterpret_cast<uint32_t*>(gRanges);
    //reinterpret_cast<volatile uint32_t&>(gRangesInt[rangeIndexInt]) = value;
    atomicExch(gRangesInt + rangeIndexInt, value);
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

        //printf("[%u], P {%f, %f, %f}, D {%u, %u, %u}\n",
        //       globalId,
        //       center[0], center[1], center[2],
        //       x, y, z);

        // 64-bit morton code can only hold 21 bit for each value
        assert(x <= (1 << 21) && y <= (1 << 21) &&
               z <= (1 << 21));
        uint64_t code = MortonCode::Compose3D<uint64_t>(Vector3ui(x, y, z));
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
static void KCConstructLBVHLeaf(uint32_t& gRootIndex,
                                LBVHNode<Leaf>* gNodes,
                                Vector2ui* gRanges,
                                const Leaf* gLeafData,
                                const uint32_t* gSortedIndices,
                                const uint64_t* gSortedMortonCodes,
                                uint32_t leafCount)
{
    LBVHNode<Leaf>* gLeafNodes = gNodes;
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId >= leafCount) return;

    gLeafNodes[globalId].isLeaf = true;
    uint32_t leafIndex = gSortedIndices[globalId];
    gLeafNodes[globalId].leaf = gLeafData[leafIndex];

    Vector2ui myRange = Vector2ui(globalId);
    ChooseParentAndUpdate(gNodes, gRanges, myRange, globalId,
                            leafCount, gSortedMortonCodes, leafCount, true);

}

template <class Leaf>
__global__
static void KCConstructLBVHInter(uint32_t& gCounter,
                                 uint32_t* gFlags,
                                 uint32_t& gRootIndex,
                                 LBVHNode<Leaf>* gNodes,
                                 Vector2ui* gRanges,
                                 const uint64_t* gSortedMortonCodes,
                                 uint32_t leafCount)
{
    uint32_t internalNodeCount = leafCount - 1;
    LBVHNode<Leaf>* gInternalNodes = gNodes + leafCount;

    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId >= internalNodeCount) return;

    Vector2ui myRange = gRanges[globalId];
    // Skip work if data is not ready
    // Now we can mask threads here
    if(myRange[0] != UINT32_MAX &&
       myRange[1] != UINT32_MAX &&
       gFlags[globalId] == 0)
    {
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

        gInternalNodes[globalId].isLeaf = false;
        Vector3f infinity = Vector3f(INFINITY);
        gInternalNodes[globalId].body.aabbMin = infinity;
        gInternalNodes[globalId].body.aabbMax = -infinity;

        atomicInc(&gCounter, UINT32_MAX);
        gFlags[globalId] = 1;
    }
}

template <class Leaf>
__global__
static void KCConstructLinearBVH(uint32_t& gRootIndex,
                                 uint32_t& gCounter,
                                 uint32_t* gFlags,
                                 LBVHNode<Leaf>* gNodes,
                                 Vector2ui* gRanges,
                                 const Leaf* gLeafData,
                                 const uint32_t* gSortedIndices,
                                 const uint64_t* gSortedMortonCodes,
                                 uint32_t leafCount)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId != 0) return;

    const uint32_t threadCount = StaticThreadPerBlock1D;
    const uint32_t blockCount = (leafCount + threadCount - 1) / threadCount;

    KCConstructLBVHLeaf<<<blockCount, threadCount>>>(gRootIndex, gNodes, gRanges,
                                                     gLeafData, gSortedIndices,
                                                     gSortedMortonCodes,
                                                     leafCount);
    //cudaDeviceSynchronize();
    __syncthreads();

    uint32_t depth = 1;
    while(gCounter < (leafCount - 1))
    {
        KCConstructLBVHInter<<<blockCount, threadCount>>>
        (
            gCounter, gFlags,
            gRootIndex, gNodes, gRanges,
            gSortedMortonCodes,
            leafCount
        );
        __syncthreads();
        depth++;
    }
    printf("LBVH Depth %u\n", depth);
}

template <class Leaf>
__global__
static void BottomUpAABBLeaf(LBVHNode<Leaf>* gNodes,
                             Vector2ui* gLRFlags,
                             uint32_t leafCount)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId >= leafCount) return;

    uint32_t parentId = gNodes[globalId].parent;
    LBVHNode<Leaf>& parentNode = gNodes[parentId];
    uint32_t parentFlagId = parentId - leafCount;
    bool isLeft = parentNode.body.left == globalId;
    uint32_t i = (isLeft) ? 0 : 1;
    gLRFlags[parentFlagId][i] = 1;

}

template <class Leaf>
__global__
static void BottomUpAABBInter(uint32_t& gCounter,
                              uint32_t* gFlags,
                              Vector2ui* gLRFlags,
                              LBVHNode<Leaf>* gNodes,
                              const AABB3f* gLeafAABBs,
                              const uint32_t* gSortedIndices,
                              uint32_t leafCount)
{
    uint32_t internalNodeCount = leafCount - 1;
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId >= internalNodeCount) return;

    LBVHNode<Leaf>* gInternalNodes = gNodes + leafCount;

    Vector2ui leftRight = gLRFlags[globalId];
    // Skip work if data is not ready
    // Now we can mask threads here
    if(leftRight[0] != UINT32_MAX &&
       leftRight[1] != UINT32_MAX &&
       gFlags[globalId] == 0)
    {
        // My Children are ready
        uint32_t left = gInternalNodes[globalId].body.left;
        uint32_t right = gInternalNodes[globalId].body.right;
        LBVHNode<Leaf>& lNode = gNodes[left];
        LBVHNode<Leaf>& rNode = gNodes[right];

        AABB3f aabbLeft = (lNode.isLeaf) ? gLeafAABBs[gSortedIndices[left]]
                                         : AABB3f(lNode.body.aabbMin,
                                                  lNode.body.aabbMax);
        AABB3f aabbRight = (rNode.isLeaf) ? gLeafAABBs[gSortedIndices[right]]
                                          : AABB3f(rNode.body.aabbMin,
                                                   rNode.body.aabbMax);

        AABB3f aabb = aabbLeft.Union(aabbRight);
        gInternalNodes[globalId].body.aabbMin = aabb.Min();
        gInternalNodes[globalId].body.aabbMax = aabb.Max();

        // Set that your data is ready (for parent)
        uint32_t parentId = gInternalNodes[globalId].parent;
        if(parentId != UINT32_MAX)
        {
            uint32_t myId = globalId + leafCount;
            uint32_t parentFlagId = parentId - leafCount;
            LBVHNode<Leaf>& parentNode = gNodes[parentId];
            bool isLeft = parentNode.body.left == myId;
            uint32_t i = (isLeft) ? 0 : 1;
            gLRFlags[parentFlagId][i] = 1;
        }
        // Set your flags so that you don't do the work twice
        gFlags[globalId] = 1;
        atomicInc(&gCounter, UINT32_MAX);
    }
}

template <class Leaf>
__global__
static void BottomUpAABBUnion(uint32_t& gCounter,
                              uint32_t* gFlags,
                              Vector2ui* gLRFlags,
                              LBVHNode<Leaf>* gNodes,
                              const AABB3f* gLeafAABBs,
                              const uint32_t* gSortedIndices,
                              uint32_t leafCount)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId != 0) return;

    const uint32_t threadCount = StaticThreadPerBlock1D;
    const uint32_t blockCount = (leafCount + threadCount - 1) / threadCount;

    BottomUpAABBLeaf<<<blockCount, threadCount>>>(gNodes, gLRFlags,
                                                  leafCount);

    __syncthreads();
    while(gCounter < (leafCount - 1))
    {
        BottomUpAABBInter<<<blockCount, threadCount>>>
        (
            gCounter, gFlags, gLRFlags,
            gNodes, gLeafAABBs,
            gSortedIndices,
            leafCount
        );
        __syncthreads();
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
    CUDA_CHECK(cudaMemset(dHaveDupMorton, 0x00, sizeof(bool)));


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
    uint32_t* dCounter;
    uint32_t* dFlags;
    DeviceMemory rangeMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dRanges, dRootIndex, dCounter, dFlags),
                                   rangeMemory,
                                   {leafCount - 1,
                                   1, 1,
                                   leafCount - 1
                                   });
    CUDA_CHECK(cudaMemset(dRanges, 0xFF, sizeof(Vector2ui) * (leafCount - 1)));
    CUDA_CHECK(cudaMemset(dCounter, 0x00, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dFlags, 0x00, sizeof(uint32_t) * (leafCount - 1)));

    // Finally Allocate the entire node array
    LBVHNode<Leaf>* dNodes;
    GPUMemFuncs::AllocateMultiData(std::tie(dNodes),
                                   memory,
                                   {leafCount + leafCount - 1});
    CUDA_CHECK(cudaMemset(dNodes, 0x00, sizeof(LBVHNode<Leaf>) * (leafCount + leafCount - 1)));

    // Call a single kernel
    // Abuse dynamic parallelism
    gpu.ExactKC_X(0, (cudaStream_t)0, 1, 1,
                  //
                  KCConstructLinearBVH<Leaf>,
                  //
                  *dRootIndex,
                  *dCounter,
                  dFlags,
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
    // Reset Flags
    CUDA_CHECK(cudaMemset(dRanges, 0xFF, sizeof(Vector2ui) * (leafCount - 1)));
    CUDA_CHECK(cudaMemset(dCounter, 0x00, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dFlags, 0x00, sizeof(uint32_t) * (leafCount - 1)));

    // Call a single kernel
    // Abuse dynamic parallelism
    gpu.ExactKC_X(0, (cudaStream_t)0, 1, 1,
                  //
                  BottomUpAABBUnion<Leaf>,
                  //
                  *dCounter,
                  dFlags,
                  dRanges,
                  dNodes,
                  dLeafAABBs,
                  dSortedIndices,
                  leafCount);

    treeGPU.rootIndex = hRootIndex;
    treeGPU.nodes = dNodes;
    treeGPU.nodeCount = leafCount + leafCount - 1;
    treeGPU.leafCount = leafCount;
    treeGPU.DistanceFunction = df;

    return TracerError::OK;
}

template <class Leaf, class DF,
          AABBGenFunc<Leaf> AF>
TracerError LinearBVHCPU<Leaf, DF, AF>::ConstructNonLinear(const Leaf* dLeafList,
                                                           uint32_t leafCount,
                                                           DF df,
                                                           const CudaSystem& system)
{
    static constexpr uint32_t MAX_BASE_DEPTH = 64;
    enum SplitAxis
    {
        X,
        Y,
        Z,
        END
    };

    // Partition Function
    auto GenBVHNode = [](// Output
                         LBVHNode<Leaf>& node,
                         size_t& splitLoc,
                         // I-O
                         std::vector<Leaf>& leafs,
                         std::vector<AABB3f>& aabbs,
                         // Args
                         uint32_t parentIndex,
                         SplitAxis axis,
                         size_t start, size_t end)
    {
        int axisIndex = static_cast<int>(axis);

        // Populate Node
        node.parent = parentIndex;
        node.isLeaf = false;

        // Base Case
        if(end - start == 1)
        {
            node.isLeaf = true;
            node.leaf = leafs[start];
            splitLoc = std::numeric_limits<size_t>::max();
        }
        else
        {
            AABB3f aabbUnion = NegativeAABB3;
            Vector3 center = Zero3;
            for(size_t j = start; j < end; j++)
            {
                AABB3f aabb = aabbs[j];
                aabbUnion.UnionSelf(aabb);
            }
            center = aabbUnion.Centroid();

            splitLoc = 0;
            constexpr int TOTAL_AXES = 3;
            for(int i = 0; i < TOTAL_AXES; i++)
            {
                int testAxis = axisIndex + i % TOTAL_AXES;
                // Partition wrt. avg center
                int64_t splitStart = static_cast<int64_t>(start - 1);
                int64_t splitEnd = static_cast<int64_t>(end);
                while(splitStart < splitEnd)
                {
                    // Hoare Like Partition
                    float leftTriAxisCenter;
                    do
                    {
                        if(splitStart >= static_cast<int64_t>(end - 1)) break;
                        splitStart++;
                        leftTriAxisCenter = aabbs[splitStart].Centroid()[testAxis];
                    }
                    while(leftTriAxisCenter >= center[testAxis]);
                    float rightTriAxisCenter;
                    do
                    {
                        if(splitEnd <= static_cast<int64_t>(start + 1)) break;
                        splitEnd--;
                        rightTriAxisCenter = aabbs[splitEnd].Centroid()[testAxis];
                    }
                    while(rightTriAxisCenter <= center[testAxis]);

                    if(splitStart < splitEnd)
                    {
                        std::swap(aabbs[splitEnd], aabbs[splitStart]);
                        std::swap(leafs[splitEnd], leafs[splitStart]);
                    }
                }

                // Test this split
                if(splitStart != static_cast<int64_t>(start) ||
                   splitStart != static_cast<int64_t>(end))
                {
                    // This is a good split save and break
                    splitLoc = splitStart;
                    break;
                }
            }
            // If cant find any proper split
            // Just cut in half
            if(splitLoc == 0) splitLoc = (end - start) / 2;

            // Sanity Check
            assert(splitLoc != start);
            assert(splitLoc != end);

            // Save AABB
            node.body.aabbMin = aabbUnion.Min();
            node.body.aabbMax = aabbUnion.Max();
        }
    };

    auto DetermineNextSplit = [](SplitAxis split, const AABB3f& aabb)
    {
        SplitAxis nextSplit = static_cast<SplitAxis>((static_cast<int>(split) + 1) %
                                                     static_cast<int>(SplitAxis::END));
        int splitIndex = static_cast<int>(nextSplit);
        // Skip this split if it is very tight (compared to other axis)
        Vector3 diff = aabb.Max() - aabb.Min();
        // AABB is like a 2D AABB skip this axis
        if(std::abs(diff[splitIndex]) < 0.001f)
            nextSplit = static_cast<SplitAxis>((static_cast<int>(nextSplit) + 1) %
                                               static_cast<int>(SplitAxis::END));
        return nextSplit;
    };

    // Load Leafs to Memory
    std::vector<Leaf> hLeafs(leafCount);
    CUDA_CHECK(cudaMemcpy(hLeafs.data(), dLeafList,
                          sizeof(Leaf) * leafCount,
                          cudaMemcpyDeviceToHost));

    // Gen AABBs
    std::vector<AABB3f> hAABBs(leafCount);
    uint32_t i = 0;
    for(AABB3f& aabb : hAABBs)
    {
        aabb = AABBFunc(hLeafs[i]);
        i++;
    }
    // CPU Memory
    std::vector<LBVHNode<Leaf>> bvhNodes;
    //
    struct SplitWork
    {
        bool left;
        size_t start;
        size_t end;
        SplitAxis axis;
        uint32_t parentId;
        uint32_t depth;
    };

    // Start Partitioning
    std::queue<SplitWork> partitionQueue;
    partitionQueue.emplace(SplitWork
                           {
                               false,
                               0, leafCount,
                               SplitAxis::X,
                               std::numeric_limits<uint32_t>::max(),
                               0
                           });

    // Breath first tree generation (top-down)
    uint8_t maxDepth = 0;
    while(!partitionQueue.empty())
    {
        SplitWork current = partitionQueue.front();
        partitionQueue.pop();

        size_t splitLoc;
        LBVHNode<Leaf> node;
        // Do Generation
        GenBVHNode(node,
                   splitLoc,
                   // I-O
                   hLeafs,
                   hAABBs,
                    // Args
                   current.parentId,
                   current.axis,
                   current.start, current.end);

        bvhNodes.emplace_back(node);
        uint32_t nextParentId = static_cast<uint32_t>(bvhNodes.size() - 1);
        SplitAxis nextSplit = DetermineNextSplit(current.axis,
                                                 AABB3(node.body.aabbMin,
                                                       node.body.aabbMax));

        // Update parent
        if(current.parentId != std::numeric_limits<uint32_t>::max())
        {
            if(current.left) bvhNodes[current.parentId].body.left = nextParentId;
            else bvhNodes[current.parentId].body.right = nextParentId;
        }

        // Check if not base case and add more generation
        if(splitLoc != std::numeric_limits<size_t>::max())
        {
            partitionQueue.emplace(SplitWork{true, current.start, splitLoc, nextSplit, nextParentId, current.depth + 1});
            partitionQueue.emplace(SplitWork{false, splitLoc, current.end, nextSplit, nextParentId, current.depth + 1});
            maxDepth = static_cast<uint8_t>(current.depth + 1);

            if((current.depth + 1) > MAX_BASE_DEPTH)
                return TracerError::TRACER_INTERNAL_ERROR;
        }
    }
    // BVH cannot hold this surface return error
    if(maxDepth > MAX_BASE_DEPTH)
        return TracerError::TRACER_INTERNAL_ERROR;

        // Finally Allocate the entire node array
    LBVHNode<Leaf>* dNodes;
    GPUMemFuncs::AllocateMultiData(std::tie(dNodes),
                                   memory,
                                   {bvhNodes.size()});
    CUDA_CHECK(cudaMemcpy(dNodes, bvhNodes.data(),
                          sizeof(LBVHNode<Leaf>)* bvhNodes.size(),
                          cudaMemcpyHostToDevice));

    treeGPU.rootIndex = 0;
    treeGPU.nodes = dNodes;
    treeGPU.nodeCount = static_cast<uint32_t>(bvhNodes.size());
    treeGPU.leafCount = leafCount;
    treeGPU.DistanceFunction = df;
    return TracerError::OK;
}

template <class Leaf, class DF,
          AABBGenFunc<Leaf> AF>
void LinearBVHCPU<Leaf, DF, AF>::DumpTreeAsBinary(std::vector<Byte>& data) const
{
    size_t bvhSize = sizeof(LBVHNode<Leaf>);
    // Allocate total size
    size_t totalSize = (sizeof(uint32_t) + // root index
                        sizeof(uint32_t) + // node count
                        sizeof(uint32_t) + // leaf count
                        treeGPU.nodeCount * bvhSize);

    data.resize(totalSize);
    Byte* dataPtr = data.data();
    // Copy
    std::memcpy(dataPtr, &treeGPU.rootIndex, sizeof(uint32_t));
    dataPtr += sizeof(uint32_t);
    std::memcpy(dataPtr, &treeGPU.nodeCount, sizeof(uint32_t));
    dataPtr += sizeof(uint32_t);
    std::memcpy(dataPtr, &treeGPU.leafCount, sizeof(uint32_t));
    dataPtr += sizeof(uint32_t);

    // Copy the rest from the GPU
    CUDA_CHECK(cudaMemcpy(dataPtr, treeGPU.nodes,
                          bvhSize * treeGPU.nodeCount,
                          cudaMemcpyDeviceToHost));
}