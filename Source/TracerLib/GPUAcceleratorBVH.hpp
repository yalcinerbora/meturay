
template <class PGroup>
GPUAccBVHGroup<PGroup>::GPUAccBVHGroup(const GPUPrimitiveGroupI& pGroup,
                                       const TransformStruct* dInvTransforms)
    : GPUAcceleratorGroup<PGroup>(pGroup, dInvTransforms)
    , dBVHLists(nullptr)    
{}

template <class PGroup>
const char* GPUAccBVHGroup<PGroup>::Type() const
{
    return TypeName();
}

template <class PGroup>
HitKey GPUAccBVHGroup<PGroup>::FindHitKey(uint32_t accIndex,
                                          PrimitiveId id)
{
    const PrimitiveRangeList& pRanges = primitiveRanges[accIndex];
    for(uint32_t i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
    {
        const auto& range = pRanges[i];
        if(range[0] == std::numeric_limits<uint64_t>::max())
            break;

        if(id >= range[0] &&
           id < range[1])
            return primitiveMaterialKeys[accIndex][i];
    }
    return HitKey::InvalidKey;
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::ConstBVHRecursive(// Output
                                               uint32_t& genNodeIndex,
                                               uint32_t parentIndex,
                                               //Temp Memory
                                               void* dTemp,
                                               void* dReduceOut,
                                               // Index Data
                                               uint32_t* dIndicesOut,
                                               uint32_t* dIndicesIn,
                                               // Constants
                                               const uint64_t* dPrimIds,
                                               const Vector3f* dPrimCenters,
                                               const AABB3f* dAABBs,
                                               uint32_t accIndex,
                                               // Call Related Args
                                               std::vector<BVHNode<LeafData>>& nodeList,
                                               SplitAxis axis,
                                               size_t start, size_t end)
{
    using PrimitiveData = typename PGroup::PrimitiveData;    
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);

    SplitAxis nextSplitAxis = static_cast<SplitAxis>((static_cast<int>(axis) + 1) %
                                                     static_cast<int>(SplitAxis::END));
    // Generate Your own Node
    // Get your node index
    nodeList.emplace_back();
    BVHNode<LeafData>& myNode = nodeList.back();
    uint32_t myIndex = static_cast<uint32_t>(nodeList.size() - 1);
    myNode.parent = parentIndex;

    size_t splitPoint;

    // Base Case (CPU Mode)
    if(end - start == 1)
    {
        PrimitiveId id = dPrimIds[start];        
        HitKey matKey = FindHitKey(accIndex, id);
        
        myNode.isLeaf = true;
        myNode.leaf = PGroup::LeafFunc(matKey, id, primData);
    }
    else if(end - start <= Threshold_CPU_GPU)
    {
        // CPU Mode
    }
    else
    {
        // GPU Mode
    }

    // Generate Child Nodex
    uint32_t indexLeft;
    uint32_t indexRight;
    // While Calling dont forget to exchange in/out
    ConstBVHRecursive(indexLeft,
                      //
                      myIndex,
                      //Temp Memory
                      dTemp,
                      dReduceOut,
                      // Index Data
                      dIndicesIn,
                      dIndicesOut,
                      // Constants
                      dPrimIds,
                      dPrimCenters,
                      dAABBs,
                      accIndex,
                      // Call Related Args
                      nodeList,
                      nextSplitAxis,
                      start, splitPoint);
    ConstBVHRecursive(indexRight,
                      //
                      myIndex,
                      //Temp Memory
                      dTemp,
                      dReduceOut,
                       // Index Data
                      dIndicesIn,
                      dIndicesOut,
                      // Constants
                      dPrimIds,
                      dPrimCenters,
                      dAABBs,
                      accIndex,
                      // Call Related Args
                      nodeList,
                      nextSplitAxis,
                      splitPoint, end);

    // Attach newly found nodes
    myNode.left = indexLeft;
    myNode.right = indexRight;
}

template <class PGroup>
SceneError GPUAccBVHGroup<PGroup>::InitializeGroup(// Map of hit keys for all materials
                                                   // w.r.t matId and primitive type
                                                   const std::map<TypeIdPair, HitKey>& allHitKeys,
                                                   // List of surface/material
                                                   // pairings that uses this accelerator type
                                                   // and primitive type
                                                   const std::map<uint32_t, IdPairs>& pairingList,
                                                   double time)
{

    const char* primGroupTypeName = primitiveGroup.Type();

    // Iterate over pairings
    int j = 0;
    for(const auto& pairings : pairingList)
    {
        PrimitiveRangeList primRangeList;
        HitKeyList hitKeyList;
        primRangeList.fill(Vector2ul(std::numeric_limits<uint64_t>::max()));
        hitKeyList.fill(HitKey::InvalidKey);

        const IdPairs& pList = pairings.second;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& p = pList[i];
            if(p.first == std::numeric_limits<uint32_t>::max())
                break;

            primRangeList[i] = primitiveGroup.PrimitiveBatchRange(p.second);
            hitKeyList[i] = allHitKeys.at(std::make_pair(primGroupTypeName, p.first));
        }
        // Put generated AABB
        primitiveRanges.push_back(primRangeList);
        primitiveMaterialKeys.push_back(hitKeyList);
        idLookup.emplace(pairings.first, j);
        j++;
    }
    assert(primitiveRanges.size() == primitiveMaterialKeys.size());
    assert(primitiveMaterialKeys.size() == idLookup.size());
    return SceneError::OK;
}

template <class PGroup>
SceneError GPUAccBVHGroup<PGroup>::ChangeTime(// Map of hit keys for all materials
                                              // w.r.t matId and primitive type
                                              const std::map<TypeIdPair, HitKey>&,
                                              // List of surface/material
                                              // pairings that uses this accelerator type
                                              // and primitive type
                                              const std::map<uint32_t, IdPairs>& pairingList,
                                              double time)
{
    // TODO:
    return SceneError::OK;
}

template <class PGroup>
uint32_t GPUAccBVHGroup<PGroup>::InnerId(uint32_t surfaceId) const
{
    return idLookup.at(surfaceId);
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::ConstructAccelerators(const CudaSystem& system)
{
    // TODO: make this a single KC
    for(const auto& id : idLookup)
    {
        ConstructAccelerator(id.first, system);
    }
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::ConstructAccelerator(uint32_t surface,
                                                  const CudaSystem& system)
{   
    using PrimitiveData = typename PGroup::PrimitiveData;
    using PrimitiveIndexStart = std::array<uint64_t, SceneConstants::MaxPrimitivePerSurface>;
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);

    uint32_t index = idLookup.at(surface);
    const PrimitiveRangeList& primRangeList = primitiveRanges[index];
    
    size_t currentOffset = 0;
    PrimitiveRangeList indexOffsets;
    for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
    {
        const auto& range = primRangeList[i];
        if(range[0] == std::numeric_limits<uint64_t>::max())
            break;

        indexOffsets[i][0] = currentOffset;
        currentOffset = range[1] - range[0];
        indexOffsets[i][1] = currentOffset;
    }
    size_t totalPrimCount = currentOffset;

    // Determine Partition/Reduce Memories
    size_t cubIfMemSize = 0;
    uint64_t* in = nullptr;
    uint64_t* out = nullptr;
    uint64_t* count = nullptr;
    CUDA_CHECK(cub::DevicePartition::If(nullptr, cubIfMemSize,
                                        in, out, count,
                                        static_cast<int>(totalPrimCount),
                                        SpacePartition<PGroup>(0, SplitAxis::X, primData)));
    size_t cubAABBReduceMemSize = 0;
    AABB3f* inAABB = nullptr;
    AABB3f* outAABB = nullptr;
    CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr,
                                         cubAABBReduceMemSize,
                                         inAABB, outAABB,
                                         static_cast<int>(totalPrimCount),
                                         AABBUnion(),
                                         AABB3f(Zero3, Zero3)));
    size_t cubCenterSumMemSize = 0;
    float* inCent = nullptr;
    float* outCent = nullptr;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr,
                                      cubCenterSumMemSize,
                                      inCent, outCent,
                                      static_cast<int>(totalPrimCount)));

    // Combine Temp Memory
    size_t tempMemSize = std::max(cubAABBReduceMemSize,
                                  std::max(cubCenterSumMemSize,
                                           cubIfMemSize));

    // GPU Memory
    size_t totalSize = totalPrimCount * (sizeof(uint64_t) +
                                         2 * sizeof(uint32_t) +
                                         sizeof(AABB3f) +
                                         sizeof(Vector3f)) +
                       tempMemSize + 
                       sizeof(AABB3f);
    DeviceMemory memory = DeviceMemory(totalSize);
    Byte* memPtr = static_cast<Byte*>(memory);
    size_t offset = 0;
    Vector3f* dPrimCenters = reinterpret_cast<Vector3f*>(memPtr + offset);
    offset += totalPrimCount * sizeof(Vector3f);
    AABB3f* dPrimAABBs = reinterpret_cast<AABB3f*>(memPtr + offset);
    offset += totalPrimCount * sizeof(AABB3f);
    uint64_t* dPrimIds = reinterpret_cast<uint64_t*>(memPtr + offset);
    offset += totalPrimCount * sizeof(uint64_t);
    uint32_t* dIds0 = reinterpret_cast<uint32_t*>(memPtr + offset);
    offset += totalPrimCount * sizeof(uint32_t);
    uint32_t* dIds1 = reinterpret_cast<uint32_t*>(memPtr + offset);
    offset += totalPrimCount * sizeof(uint32_t);
    void* dTemp = reinterpret_cast<void*>(memPtr + offset);
    offset += tempMemSize;
    void* dReduceOut = reinterpret_cast<void*>(memPtr + offset);
    offset += sizeof(AABB3f);
    assert(offset == totalSize);

    // Populate Memory
    const CudaGPU& gpu = (*system.GPUList().begin());    
    // Populate Indices
    size_t indexOffset = 0;
    for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
    {
        const auto& range = primRangeList[i];
        if(range[0] == std::numeric_limits<uint64_t>::max())
            break;

        uint32_t indexCount = static_cast<uint32_t>(range[1] - range[0]);

        gpu.GridStrideKC_X(0, 0,
                           indexCount,
                           //
                           KCInitIndices,
                           //
                           dIds0 + indexOffset,
                           dPrimIds + indexOffset,
                           //
                           static_cast<uint32_t>(indexOffset),
                           range[0],
                           indexCount);

        indexOffset += indexCount;
    }

    // Now Populate AABB and Center Points
    gpu.GridStrideKC_X(0, 0,
                       totalPrimCount,
                       //
                       KCGenCenters<PGroup>,
                       //
                       dPrimCenters,
                       //
                       dIds0,
                       dPrimIds,
                       //
                       CentroidGen<PGroup>(primData),
                       static_cast<uint32_t>(totalPrimCount));
    //
    gpu.GridStrideKC_X(0, 0,
                       totalPrimCount,
                       //
                       KCGenAABBs<PGroup>,
                       //
                       dPrimAABBs,
                       //
                       dIds0,
                       dPrimIds,
                       //
                       AABBGen<PGroup>(primData),
                       static_cast<uint32_t>(totalPrimCount));

    // CPU Memory
    std::vector<BVHNode<LeafData>>& bvhNodes;
    
    // Now ready for recursion
    uint32_t rootIndex;
    ConstBVHRecursive(// Output
                      rootIndex,
                      //
                      std::numeric_limits<uint32_t>::max(),
                      //Temp Memory                      
                      dTemp,
                      dReduceOut,
                      // Index Data
                      dIds1,
                      dIds0,
                      // Constants
                      dPrimIds,
                      dPrimCenters,
                      dPrimAABBs,
                      index,
                      // Call Related Args
                      bvhNodes,
                      SplitAxis::X,
                      0, totalPrimCount);

    // Finally Nodes are Generated now copy it to GPU Memory
    ...

}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::ConstructAccelerators(const std::vector<uint32_t>& surfaces,
                                                   const CudaSystem& system)
{
    // TODO: make this a single KC
    for(const uint32_t& id : surfaces)
    {
        ConstructAccelerator(id, system);
    }
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::DestroyAccelerators(const CudaSystem&)
{
    //...
    // Define destory??
    // There is no destruction or deallocation
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::DestroyAccelerator(uint32_t surface,
                                                const CudaSystem&)
{
    //...
    // Define destory??
    // There is no destruction or deallocation
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::DestroyAccelerators(const std::vector<uint32_t>& surfaces,
                                                 const CudaSystem&)
{
    //...
}

template <class PGroup>
size_t GPUAccBVHGroup<PGroup>::UsedGPUMemory() const
{
    return memory.Size();
}

template <class PGroup>
size_t GPUAccBVHGroup<PGroup>::UsedCPUMemory() const
{
    // TODO:
    // Write allocator wrapper for which keeps track of total bytes allocated
    // and deallocated
    return 0;
}

template <class PGroup>
GPUAccBVHBatch<PGroup>::GPUAccBVHBatch(const GPUAcceleratorGroupI& a,
                                       const GPUPrimitiveGroupI& p)
    : GPUAcceleratorBatch(a, p)
{}

template <class PGroup>
const char* GPUAccBVHBatch<PGroup>::Type() const
{
    return TypeName();
}

template <class PGroup>
void GPUAccBVHBatch<PGroup>::Hit(const CudaGPU& gpu,
                                 // O
                                 HitKey* dMaterialKeys,
                                 PrimitiveId* dPrimitiveIds,
                                 HitStructPtr dHitStructs,
                                 // I-O                                                  
                                 RayGMem* dRays,
                                 // Input
                                 const TransformId* dTransformIds,
                                 const RayId* dRayIds,
                                 const HitKey* dAcceleratorKeys,
                                 const uint32_t rayCount) const
{
    // TODO: Is there a better way to implement this
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);

    gpu.AsyncGridStrideKC_X
    (
        0,
        rayCount,
        //
        KCIntersectBVH<PGroup>,
        // Args
        // O
        dMaterialKeys,
        dPrimitiveIds,
        dHitStructs,
        // I-O
        dRays,
        // Input
        dTransformIds,
        dRayIds,
        dAcceleratorKeys,
        rayCount,
        // Constants
        acceleratorGroup.dBVHLists,
        acceleratorGroup.dInverseTransforms,
        //
        primData
    );
}