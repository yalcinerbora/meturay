template <class PGroup>
GPUAccBVHGroup<PGroup>::GPUAccBVHGroup(const GPUPrimitiveGroupI& pGroup)
    : GPUAcceleratorGroup<PGroup>(pGroup)
    , dBVHLists(nullptr)
{}

template <class PGroup>
const char* GPUAccBVHGroup<PGroup>::Type() const
{
    return TypeName();
}

template <class PGroup>
HitKey GPUAccBVHGroup<PGroup>::FindHitKey(uint32_t accIndex,
                                          PrimitiveId id,
                                          bool doKeyExpand)
{
    const PrimitiveRangeList& pRanges = primitiveRanges[accIndex];
    for(uint32_t i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
    {
        const auto& range = pRanges[i];
        if(range[0] == std::numeric_limits<uint64_t>::max())
            break;

        if(id >= range[0] && id < range[1])
        {
            HitKey key = primitiveMaterialKeys[accIndex][i];
            if(!doKeyExpand) return key;

            PrimitiveId expansion = id - range[0];
            PrimitiveId expandedId = HitKey::FetchIdPortion(key) + expansion;
            key = HitKey::CombinedKey(HitKey::FetchBatchPortion(key),
                                        static_cast<HitKey::Type>(expandedId));
            return key;
        }
    }
    return HitKey::InvalidKey;
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::GenerateBVHNode(// Output
                                             size_t& splitLoc,
                                             BVHNode<LeafData>& node,
                                             //Temp Memory
                                             void* dTemp,
                                             size_t tempMemSize,
                                             uint32_t* dPartitionSplitOut,
                                             uint32_t* dIndicesTemp,
                                             // Index Data
                                             uint32_t* dIndicesIn,
                                             // Constants
                                             const uint64_t* dPrimIds,
                                             const Vector3f* dPrimCenters,
                                             const AABB3f* dAABBs,
                                             uint32_t accIndex,
                                             bool doKeyExpand,
                                             const CudaGPU& gpu,
                                             // Call Related Args
                                             uint32_t parentIndex,
                                             SplitAxis axis,
                                             size_t start, size_t end)
{
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);

    int axisIndex = static_cast<int>(axis);

    // Populate Node
    node.parent = parentIndex;
    node.isLeaf = false;

    // Base Case (CPU Mode)
    if(end - start == 1)
    {
        uint32_t index = dIndicesIn[start];
        PrimitiveId id = dPrimIds[index];
        HitKey matKey = FindHitKey(accIndex, id, doKeyExpand);

        node.isLeaf = true;
        node.leaf = PGroup::Leaf(matKey, id, primData);
        splitLoc = std::numeric_limits<size_t>::max();
    }
    else if(end - start <= Threshold_CPU_GPU)
    {
        splitLoc = 0;
        AABB3f aabbUnion = NegativeAABB3;
        Vector3 avgCenter = Zero3;

        constexpr int TOTAL_AXES = 3;
        for(int i = 0; i < TOTAL_AXES; i++)
        {
            int testAxis = axisIndex + i % TOTAL_AXES;
            // Find AABB and Center
            // Do it only once
            if(i == 0)
            {
                for(size_t j = start; j < end; j++)
                {
                    uint32_t index = dIndicesIn[j];

                    AABB3f aabb = dAABBs[index];
                    aabbUnion.UnionSelf(aabb);

                    Vector3 center = dPrimCenters[index];
                    float size = static_cast<float>(j - start);
                    avgCenter = (avgCenter * size + center) / (size + 1.0f);
                }
            }

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

                    uint32_t index = dIndicesIn[splitStart];
                    leftTriAxisCenter = dPrimCenters[index][testAxis];
                } while(leftTriAxisCenter >= avgCenter[testAxis]);
                float rightTriAxisCenter;
                do
                {
                    if(splitEnd <= static_cast<int64_t>(start + 1)) break;
                    splitEnd--;
                    uint32_t index = dIndicesIn[splitEnd];
                    rightTriAxisCenter = dPrimCenters[index][testAxis];
                } while(rightTriAxisCenter <= avgCenter[testAxis]);

                if(splitStart < splitEnd)
                    std::swap(dIndicesIn[splitEnd], dIndicesIn[splitStart]);
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
    else
    {
        AABB3f aabb;
        float center;

        // Find Reduced Center and AABB Union
        size_t reductionCount = (end - start);
        const size_t offset0 = (reductionCount + StaticThreadPerBlock1D - 1) / StaticThreadPerBlock1D;

        // Calculate Temp Memory
        AABB3f* aabbTemp0 = static_cast<AABB3f*>(dTemp);
        AABB3f* aabbTemp1 = static_cast<AABB3f*>(dTemp) + offset0;
        do
        {
            if(reductionCount == (end - start))
                gpu.KC_X(0, (cudaStream_t)0, reductionCount,
                         KCReduceAABBsFirst,
                         //
                         aabbTemp0,
                         dAABBs,
                         dIndicesIn + start,
                         static_cast<uint32_t>(reductionCount));
            else
            {
                gpu.KC_X(0, (cudaStream_t)0, reductionCount,
                         KCReduceAABBs,
                         //
                         aabbTemp1,
                         aabbTemp0,
                         static_cast<uint32_t>(reductionCount));
                std::swap(aabbTemp0, aabbTemp1);
            }

            reductionCount = (reductionCount + StaticThreadPerBlock1D - 1) / StaticThreadPerBlock1D;
        } while(reductionCount != 1);

        // Copy Host
        CUDA_CHECK(cudaMemcpy(&aabb, aabbTemp0, sizeof(AABB3f), cudaMemcpyDeviceToHost));

        // Now do for centroid
        reductionCount = (end - start);
        // Calculate Temp Memory
        float* centerTemp0 = static_cast<float*>(dTemp);
        float* centerTemp1 = static_cast<float*>(dTemp) + offset0;
        do
        {
            if(reductionCount == (end - start))
                gpu.KC_X(0, (cudaStream_t)0, reductionCount,
                         KCReduceCentroidsFirst,
                         //
                         centerTemp0,
                         dPrimCenters,
                         dIndicesIn + start,
                         axis,
                         static_cast<uint32_t>(reductionCount));
            else
            {
                gpu.KC_X(0, (cudaStream_t)0, reductionCount,
                         KCReduceCentroids,
                         //
                         centerTemp1,
                         centerTemp0,
                         static_cast<uint32_t>(reductionCount));
                std::swap(centerTemp0, centerTemp1);
            }
            reductionCount = (reductionCount + StaticThreadPerBlock1D - 1) / StaticThreadPerBlock1D;
        } while(reductionCount != 1);

        // Copy to Host
        CUDA_CHECK(cudaMemcpy(&center, centerTemp0, sizeof(float), cudaMemcpyDeviceToHost));
        center /= static_cast<float>((end - start));

        // Now Do Partition
        CUDA_CHECK(cub::DevicePartition::If(dTemp, tempMemSize,
                                            dIndicesIn + start,
                                            dIndicesTemp + start,
                                            dPartitionSplitOut,
                                            static_cast<int>(end - start),
                                            SpacePartition(center, axis, dPrimCenters)));

        uint32_t partitionSplit;
        CUDA_CHECK(cudaMemcpy(dIndicesIn + start,
                              dIndicesTemp + start,
                              static_cast<int>(end - start) * sizeof(uint32_t),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(&partitionSplit, dPartitionSplitOut, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // If there is bad partition (location is start or end) then at least put single node to a split
        if(partitionSplit == 0) partitionSplit += 1;
        else if(partitionSplit == static_cast<uint32_t>(end - start)) partitionSplit -= 1;
        // Split Loc
        splitLoc = partitionSplit + start;

        // Init Nodes
        node.body.aabbMin = aabb.Min();
        node.body.aabbMax = aabb.Max();
    }
}

template <class PGroup>
SceneError GPUAccBVHGroup<PGroup>::InitializeGroup(// Accelerator Option Node
                                                   const SceneNodePtr& node,
                                                   // List of surface/material
                                                   // pairings that uses this accelerator type
                                                   // and primitive type
                                                   const std::map<uint32_t, SurfaceDefinition>& surfaceList,
                                                   double)
{
    //const char* primGroupTypeName = this->primitiveGroup.Type();

    std::vector<uint32_t> hTransformIndices;
    hTransformIndices.reserve(surfaceList.size());
    surfaceLeafCounts.reserve(surfaceList.size());
    bvhNodeCounts.reserve(surfaceList.size());

    // Get params
    bool useStack = node->CommonBool(USE_STACK_NAME);
    params.useStack = useStack;

    // Iterate over pairings
    int surfaceInnerId = 0;
    for(const auto& surface : surfaceList)
    {
        PrimitiveRangeList primRangeList;
        HitKeyList hitKeyList;
        primRangeList.fill(Vector2ul(std::numeric_limits<uint64_t>::max()));
        hitKeyList.fill(HitKey::InvalidKey);

        size_t totalPrimCountInSurface = 0;
        const IdKeyPairs& pList = surface.second.primIdWorkKeyPairs;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const IdKeyPair& p = pList[i];
            if(p.first == std::numeric_limits<uint32_t>::max())
                break;

            primRangeList[i] = this->primitiveGroup.PrimitiveBatchRange(p.first);
            hitKeyList[i] = p.second;
            totalPrimCountInSurface += primRangeList[i][1] - primRangeList[i][0];
        }

        surfaceLeafCounts.push_back(static_cast<uint32_t>(totalPrimCountInSurface));
        hTransformIndices.push_back(surface.second.globalTransformIndex);
        primitiveRanges.push_back(primRangeList);
        primitiveMaterialKeys.push_back(hitKeyList);
        idLookup.emplace(surface.first, surfaceInnerId);
        keyExpandOption.push_back(surface.second.doKeyExpansion);
        surfaceInnerId++;
    }
    assert(surfaceInnerId == hTransformIndices.size());

    // Allocate device memory for BVH root ptrs
    // and transform indices
    uint32_t bvhCount = surfaceInnerId;
    // Allocate Empty Device memory Objects
    bvhDepths.resize(bvhCount, 0);
    bvhMemories.resize(bvhCount);

    // Finally Allocate and load to GPU memory
    GPUMemFuncs::AllocateMultiData(std::tie(dAccTransformIds, dBVHLists),
                                   memory,
                                   {bvhCount, bvhCount});

    // Copy Transforms
    CUDA_CHECK(cudaMemcpy(dAccTransformIds,
                          hTransformIndices.data(),
                          bvhCount * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    assert(primitiveRanges.size() == primitiveMaterialKeys.size());
    assert(primitiveMaterialKeys.size() == idLookup.size());
    return SceneError::OK;
}

template <class PGroup>
uint32_t GPUAccBVHGroup<PGroup>::InnerId(uint32_t surfaceId) const
{
    return idLookup.at(surfaceId);
}

template <class PGroup>
TracerError GPUAccBVHGroup<PGroup>::ConstructAccelerators(const CudaSystem& system)
{
    // TODO: make this a single KC
    TracerError e = TracerError::OK;
    for(const auto& id : idLookup)
    {
        if((e = ConstructAccelerator(id.first, system)) != TracerError::OK)
            return e;
    }
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccBVHGroup<PGroup>::ConstructAccelerator(uint32_t surface,
                                                         const CudaSystem& system)
{
    Utility::CPUTimer t;
    t.Start();

    using PrimitiveData = typename PGroup::PrimitiveData;
    using PrimitiveIndexStart = std::array<uint64_t, SceneConstants::MaxPrimitivePerSurface>;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);

    // Set Device
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));

    uint32_t innerIndex = idLookup.at(surface);
    const PrimitiveRangeList& primRangeList = primitiveRanges[innerIndex];
    const PrimTransformType tType = this->primitiveGroup.TransformType();

    // Select Transform for construction
    const GPUTransformI* worldTransform = nullptr;
    AcquireAcceleratorGPUTransform(worldTransform,
                                   dAccTransformIds,
                                   this->dTransforms,
                                   innerIndex,
                                   gpu);
    const GPUTransformI* transform = worldTransform;
    if(tType == PrimTransformType::CONSTANT_LOCAL_TRANSFORM)
    {
        AcquireIdentityTransform(transform,
                                 this->dTransforms,
                                 this->identityTransformIndex,
                                 gpu);
    }

    size_t currentOffset = 0;
    PrimitiveRangeList indexOffsets;
    for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
    {
        const auto& range = primRangeList[i];
        if(range[0] == std::numeric_limits<uint64_t>::max())
            break;

        indexOffsets[i][0] = currentOffset;
        currentOffset += (range[1] - range[0]);
        indexOffsets[i][1] = currentOffset;
    }
    size_t totalPrimCount = currentOffset;

    // Determine Partition / Reduce Memories
    size_t cubIfMemSize = 0;
    uint64_t* in = nullptr;
    uint64_t* out = nullptr;
    uint64_t* count = nullptr;
    CUDA_CHECK(cub::DevicePartition::If(nullptr, cubIfMemSize,
                                        in, out, count,
                                        static_cast<int>(totalPrimCount),
                                        SpacePartition(0, SplitAxis::X, nullptr)));
    size_t firstSplit = (totalPrimCount + StaticThreadPerBlock1D - 1) / StaticThreadPerBlock1D;
    size_t secondSplit = (firstSplit + StaticThreadPerBlock1D - 1) / StaticThreadPerBlock1D;
    size_t aabbReduceMemSize = sizeof(AABB3f) * (firstSplit + secondSplit);
    size_t centerReduceMemSize = sizeof(float) * (firstSplit + secondSplit);

    // Combine Required Temp Memory
    size_t tempMemSize = std::max(cubIfMemSize,
                                  std::max(centerReduceMemSize,
                                           aabbReduceMemSize));
    // GPU Memory
    uint64_t*   dPrimIds;
    uint32_t*   dIdsIn;
    uint32_t*   dIdsTemp;
    AABB3f*     dPrimAABBs;
    Vector3f*   dPrimCenters;
    uint32_t*   dPartitionSplitOut;
    Byte*       dTemp;
    DeviceMemory tempMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dPrimIds, dIdsIn, dIdsTemp,
                                            dPrimAABBs, dPrimCenters,
                                            dPartitionSplitOut, dTemp),
                                   tempMemory,
                                   {totalPrimCount, totalPrimCount, totalPrimCount,
                                   totalPrimCount, totalPrimCount, 1, tempMemSize});
    // Populate Memory
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
                           KCInitPrimIdsAndIndices,
                           //
                           dIdsIn + indexOffset,
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
                       KCGenCentersWithIndex<PGroup>,
                       //
                       dPrimCenters,
                       //
                       dIdsIn,
                       dPrimIds,
                       //
                       CentroidGen<PGroup>(primData, *transform),
                       static_cast<uint32_t>(totalPrimCount));
    //
    gpu.GridStrideKC_X(0, 0,
                       totalPrimCount,
                       //
                       KCGenAABBsWithIndex<PGroup>,
                       //
                       dPrimAABBs,
                       //
                       dIdsIn,
                       dPrimIds,
                       //
                       AABBGen<PGroup>(primData, *transform),
                       static_cast<uint32_t>(totalPrimCount));

    // CPU Memory
    std::vector<BVHNode<LeafData>> bvhNodes;
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
    std::queue<SplitWork> partitionQueue;
    partitionQueue.emplace(SplitWork
                           {
                               false,
                               0, totalPrimCount,
                               SplitAxis::X,
                               std::numeric_limits<uint32_t>::max(),
                               0
                           });

    gpu.WaitMainStream();

    // TODO: this is required for cuda when accessing managed memory
    // but it shouldn't since we are syncing main stream just above
    CUDA_CHECK(cudaDeviceSynchronize());

    // Breath first tree generation (top-down)
    uint8_t maxDepth = 0;
    while(!partitionQueue.empty())
    {
        SplitWork current = partitionQueue.front();
        partitionQueue.pop();

        size_t splitLoc;
        BVHNode<LeafData> node;
        // Do Generation
        GenerateBVHNode(splitLoc,
                        node,
                        //Temp Memory
                        dTemp,
                        tempMemSize,
                        dPartitionSplitOut,
                        dIdsTemp,
                        // Index Data
                        dIdsIn,
                        // Constants
                        dPrimIds,
                        dPrimCenters,
                        dPrimAABBs,
                        innerIndex,
                        keyExpandOption[innerIndex],
                        gpu,
                        // Call Related Args
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

            if((current.depth + 1) > MAX_DEPTH)
                return TracerError::UNABLE_TO_CONSTRUCT_ACCELERATOR;
        }
    }
     // BVH cannot hold this surface return error
    if(maxDepth > MAX_DEPTH)
        return TracerError::UNABLE_TO_CONSTRUCT_ACCELERATOR;

    // Finally Nodes are Generated now copy it to GPU Memory
    bvhMemories[innerIndex] = std::move(DeviceMemory(sizeof(BVHNode<LeafData>) * bvhNodes.size()));
    bvhDepths[innerIndex] = maxDepth;

    //Debug::DumpMemToFile("BVHNodes", bvhNodes.data(), bvhNodes.size());
    //Debug::DumpMemToFile("AABB", dPrimAABBs, totalPrimCount);
    //Debug::DumpMemToFile("dPrimIds", dPrimIds, totalPrimCount);
    //Debug::DumpMemToFile("dIds", dIdsIn, totalPrimCount);
    //METU_LOG("-------");

    // Before copying get roots AABB for base accelerator

    // Check if we have a single node in the BVH
    // If yes get the AABB from the index
    AABB3f accAABB;
    if(!bvhNodes[0].isLeaf)
    {
        accAABB = AABB3f(bvhNodes[0].body.aabbMin,
                         bvhNodes[0].body.aabbMax);
    }
    else
    {
        // Directly fetch it from the GPU memory
        CUDA_CHECK(cudaDeviceSynchronize());
        accAABB = dPrimAABBs[0];
    }

    // If constant local primitive's transform requirement is
    // constant local transform,
    if(tType == PrimTransformType::CONSTANT_LOCAL_TRANSFORM)
    {
        // transform this AABB to world space
        // since base Accelerator works on world space
        TransformLocalAABBToWorld(accAABB, *worldTransform, gpu);
    }

    // Add to the list which will be delegated to the base accelerator
    surfaceAABBs.emplace(surface, accAABB);
    // Set node count of this accelerator
    bvhNodeCounts.push_back(static_cast<uint32_t>(bvhNodes.size()));

    CUDA_CHECK(cudaMemcpy(bvhMemories[innerIndex],
                          bvhNodes.data(),
                          sizeof(BVHNode<LeafData>) * bvhNodes.size(),
                          cudaMemcpyHostToDevice));

    BVHNode<LeafData>* dBVHStart = static_cast<BVHNode<LeafData>*>(bvhMemories[innerIndex]);
    CUDA_CHECK(cudaMemcpy(dBVHLists + innerIndex,
                          &dBVHStart,
                          sizeof(BVHNode<LeafData>*),
                          cudaMemcpyHostToDevice));


    t.Stop();
    METU_LOG("Surface{:d} BVH(d={:d}) generated in {:f} seconds.",
             surface,
             maxDepth,
             t.Elapsed<CPUTimeSeconds>());

    gpu.WaitMainStream();
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccBVHGroup<PGroup>::ConstructAccelerators(const std::vector<uint32_t>& surfaces,
                                                          const CudaSystem& system)
{
    // TODO: make this a single KC
    TracerError e = TracerError::OK;
    for(const uint32_t& id : surfaces)
    {
        auto it = idLookup.cend();
        if((it = idLookup.find(id)) == idLookup.cend()) continue;

        if((e = ConstructAccelerator(it->second, system)) != TracerError::OK)
            return e;
    }
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccBVHGroup<PGroup>::DestroyAccelerators(const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccBVHGroup<PGroup>::DestroyAccelerator(uint32_t,
                                                       const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccBVHGroup<PGroup>::DestroyAccelerators(const std::vector<uint32_t>&,
                                                        const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
size_t GPUAccBVHGroup<PGroup>::UsedGPUMemory() const
{
    size_t total = memory.Size();
    for(const DeviceMemory& m : bvhMemories)
    {
        total += m.Size();
    }
    return total;
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
void GPUAccBVHGroup<PGroup>::Hit(const CudaGPU& gpu,
                                 // O
                                 HitKey* dMaterialKeys,
                                 TransformId* dTransformIds,
                                 PrimitiveId* dPrimitiveIds,
                                 HitStructPtr dHitStructs,
                                 // I-O
                                 RayGMem* dRays,
                                 // Input
                                 const RayId* dRayIds,
                                 const HitKey* dAcceleratorKeys,
                                 const uint32_t rayCount) const
{
    // TODO: Is there a better way to implement this
    using LeafData = typename PGroup::LeafData;
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);

    // Select Intersection algorithm with or without stack
    using BVHIntersectKernel = void(*)(HitKey*,
                                       TransformId*,
                                       PrimitiveId*,
                                       HitStructPtr,
                                       RayGMem*,
                                       const RayId*,
                                       const HitKey*,
                                       uint32_t,
                                       const BVHNode<LeafData>**,
                                       const GPUTransformI**,
                                       const TransformId*,
                                       const PrimTransformType,
                                       PrimitiveData);

    BVHIntersectKernel kernel = (params.useStack) ? KCIntersectBVH<PGroup> :
                                                    KCIntersectBVHStackless<PGroup>;

    // Kernel Call
    gpu.AsyncGridStrideKC_X
    (
        0,
        rayCount,
        //
        *kernel,
        // Args
        // O
        dMaterialKeys,
        dTransformIds,
        dPrimitiveIds,
        dHitStructs,
        // I-O
        dRays,
        // Input
        dRayIds,
        dAcceleratorKeys,
        rayCount,
        // Constants
        dBVHLists,
        this->dTransforms,
        dAccTransformIds,
        this->primitiveGroup.TransformType(),
        //
        primData
    );
}

template <class PGroup>
const SurfaceAABBList& GPUAccBVHGroup<PGroup>::AcceleratorAABBs() const
{
    return surfaceAABBs;
}

template <class PGroup>
size_t GPUAccBVHGroup<PGroup>::AcceleratorCount() const
{
    return idLookup.size();
}

template <class PGroup>
size_t GPUAccBVHGroup<PGroup>::TotalPrimitiveCount() const
{
    uint32_t totalLeafCount = std::reduce(surfaceLeafCounts.cbegin(),
                                          surfaceLeafCounts.cend(),
                                          0u);

    return totalLeafCount;
}

template <class PGroup>
float GPUAccBVHGroup<PGroup>::TotalApproximateArea(const CudaSystem& system) const
{
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);

    const CudaGPU& gpu = system.BestGPU();
    uint32_t totalLeafCount = static_cast<uint32_t>(TotalPrimitiveCount());

    // Linearize the leafs and transform ids for all accelerators
    // In this group
    DeviceMemory areaMemory;
    float* dAreas;
    LeafData* dLinearLeafData;
    TransformId* dLeafTransformIds;
    GPUMemFuncs::AllocateMultiData(std::tie(dAreas, dLinearLeafData,
                                            dLeafTransformIds),
                                   areaMemory,
                                   {totalLeafCount, totalLeafCount,
                                   totalLeafCount});

    // Populate the leafs
    uint32_t offset = 0;
    DeviceMemory offsetMemory;
    for(uint32_t accId = 0;
        accId < static_cast<uint32_t>(idLookup.size());
        accId++)
    {
        uint32_t localNodeCount = bvhNodeCounts[accId];

        // Allocate a temp offset array
        uint32_t* dMarks = nullptr;
        uint32_t* dOffsets = nullptr;
        GPUMemFuncs::AllocateMultiData(std::tie(dMarks, dOffsets),
                                       offsetMemory,
                                       {localNodeCount, localNodeCount});

        gpu.GridStrideKC_X(0, (cudaStream_t)0, localNodeCount,
                           //
                           KCMarkLeafs<LeafData>,
                           // Output
                           dMarks,
                           // Input
                           dBVHLists,
                           accId,
                           localNodeCount);

        // Scan the find the offsets
        ExclusiveScanArrayGPU<uint32_t, ReduceAdd<uint32_t>>(dOffsets,
                                                             dMarks,
                                                             localNodeCount,
                                                             0u);

        // Write Subset of the
        gpu.GridStrideKC_X(0, (cudaStream_t)0, localNodeCount,
                           //
                           KCCopyLeafs<LeafData>,
                           // Input
                           dLinearLeafData + offset,
                           // Output
                           dOffsets,
                           dBVHLists,
                           accId,
                           localNodeCount);
        // Expand the transform over leafs
        ExpandValueGPU<TransformId>(dLeafTransformIds + offset,
                                    dAccTransformIds + accId,
                                    surfaceLeafCounts[accId]);

        offset += surfaceLeafCounts[accId];
    }
    assert(offset == totalLeafCount);

    // Generate Area of each primitive
    gpu.GridStrideKC_X(0, (cudaStream_t)0, totalLeafCount,
                       //
                       KCGenerateAreas<PGroup>,
                       // Output
                       dAreas,
                       // Input
                       dLinearLeafData,
                       dLeafTransformIds,
                       this->dTransforms,
                       // Constants
                       primData,
                       totalLeafCount);

    // Reduce the areas
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    float hAreaTotal;
    ReduceArrayGPU<float, ReduceAdd<float>, cudaMemcpyDeviceToHost>
    (
        hAreaTotal,
        dAreas,
        totalLeafCount,
        0.0f
    );
    gpu.WaitMainStream();
    return hAreaTotal;
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::SampleAreaWeightedPoints(// Outs
                                                      Vector3f* dPositions,
                                                      Vector3f* dNormals,
                                                      // I-O
                                                      RNGSobolCPU& rngCPU,
                                                      // Inputs
                                                      uint32_t surfacePatchCount,
                                                      const CudaSystem& system) const
{
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(this->primitiveGroup);

    const CudaGPU& gpu = system.BestGPU();
    uint32_t totalLeafCount = static_cast<uint32_t>(TotalPrimitiveCount());

    // Linearize the leafs and transform ids for all accelerators
    // In this group
    DeviceMemory areaMemory;
    float* dAreas;
    LeafData* dLinearLeafData;
    TransformId* dLeafTransformIds;
    GPUMemFuncs::AllocateMultiData(std::tie(dAreas, dLinearLeafData,
                                            dLeafTransformIds),
                                   areaMemory,
                                   {totalLeafCount, totalLeafCount,
                                   totalLeafCount});

    // Populate the leafs
    uint32_t offset = 0;
    DeviceMemory offsetMemory;
    for(uint32_t accId = 0;
        accId < static_cast<uint32_t>(idLookup.size());
        accId++)
    {
        uint32_t localNodeCount = bvhNodeCounts[accId];

        // Allocate a temp offset array
        uint32_t* dMarks = nullptr;
        uint32_t* dOffsets = nullptr;
        GPUMemFuncs::AllocateMultiData(std::tie(dMarks, dOffsets),
                                       offsetMemory,
                                       {localNodeCount, localNodeCount});

        gpu.GridStrideKC_X(0, (cudaStream_t)0, localNodeCount,
                           //
                           KCMarkLeafs<LeafData>,
                           // Output
                           dMarks,
                           // Input
                           dBVHLists,
                           accId,
                           localNodeCount);

        // Scan the find the offsets
        ExclusiveScanArrayGPU<uint32_t, ReduceAdd<uint32_t>>(dOffsets,
                                                             dMarks,
                                                             localNodeCount,
                                                             0u);

        // Write Subset of the
        gpu.GridStrideKC_X(0, (cudaStream_t)0, localNodeCount,
                           //
                           KCCopyLeafs<LeafData>,
                           // Input
                           dLinearLeafData + offset,
                           // Output
                           dOffsets,
                           dBVHLists,
                           accId,
                           localNodeCount);
        // Expand the transform over leafs
        ExpandValueGPU<TransformId>(dLeafTransformIds + offset,
                                    dAccTransformIds + accId,
                                    surfaceLeafCounts[accId]);

        offset += surfaceLeafCounts[accId];
    }
    assert(offset == totalLeafCount);
    // Generate Area of each primitive
    gpu.GridStrideKC_X(0, (cudaStream_t)0, totalLeafCount,
                       //
                       KCGenerateAreas<PGroup>,
                       // Output
                       dAreas,
                       // Input
                       dLinearLeafData,
                       dLeafTransformIds,
                       this->dTransforms,
                       // Constants
                       primData,
                       totalLeafCount);

    // Generate PWC Distribution over area
    std::vector<const float*> dAreaPtrs = {dAreas};
    std::vector<size_t> counts = {totalLeafCount};
    PWCDistributionGroupCPU1D areaDist(dAreaPtrs, counts, system);

    // Now use this to fetch surface patches
    gpu.GridStrideKC_X(0, (cudaStream_t)0, surfacePatchCount,
                       //
                       KCSampleSurfacePatch<PGroup, RNGSobolGPU>,
                       // Inputs
                       dPositions,
                       dNormals,
                       // I-O
                       rngCPU.GetGPUGenerators(gpu),
                       //
                       dLinearLeafData,
                       dLeafTransformIds,
                       this->dTransforms,
                       // Constants
                       areaDist.DistributionGPU(0),
                       primData,
                       surfacePatchCount);

    // All Done!
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::EachPrimVoxelCount(// Output
                                                uint64_t* dVoxCounts,
                                                // Inputs
                                                uint32_t resolutionXYZ,
                                                const AABB3f& sceneAABB,
                                                const CudaSystem& system) const
{
    METU_ERROR_LOG("Acc BVH (Old) did not implemented voxelize surfaces yet");
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::VoxelizeSurfaces(// Outputs
                                              uint64_t* dVoxels,
                                              Vector2us* dVoxelNormals,
                                              HitKey* gVoxelLightKeys,
                                              // Inputs
                                              const uint64_t* dVoxelOffsets, // For each primitive
                                              // Light Lookup Table (Binary Search)
                                              const HitKey* dLightKeys,      // Sorted
                                              uint32_t totalLightCount,
                                              // Constants
                                              uint32_t resolutionXYZ,
                                              const AABB3f& sceneAABB,
                                              const CudaSystem& system) const
{
    METU_ERROR_LOG("Acc BVH (Old) did not implemented voxelize surfaces yet");
}