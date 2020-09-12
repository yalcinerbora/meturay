
//#include "TracerDebug.h"
//#include "DefaultLeaf.h"
//
//template<class T>
//inline std::ostream& operator<<(std::ostream& stream, const BVHNode<T>& v)
//{
//    stream << std::setw(0);
//    if(v.isLeaf)
//    {
//        stream << "leaf - " << v.leaf;
//    }
//    else
//    {
//        stream << "[ " << v.left << ", " << v.right << ", " << v.parent << " ]";
//        stream << AABB3(v.aabbMin, v.aabbMax);
//    }
//    return stream;
//}
//
//inline std::ostream& operator<<(std::ostream& stream, const BaseLeaf& l)
//{
//    stream << "[ " << l.accKey << ", " << l.transformId << " ]";
//    stream << AABB3(l.aabbMin, l.aabbMax);
//    return stream;
//}

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
                                             const CudaGPU& gpu,
                                             // Call Related Args
                                             uint32_t parentIndex,
                                             SplitAxis axis,
                                             size_t start, size_t end)
{
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);

    int axisIndex = static_cast<int>(axis);

    // Populate Node
    node.parent = parentIndex;
    node.isLeaf = false;

    // Base Case (CPU Mode)
    if(end - start == 1)
    {
        uint32_t index = dIndicesIn[start];
        PrimitiveId id = dPrimIds[index];
        HitKey matKey = FindHitKey(accIndex, id);

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
            if(splitStart != start ||
               splitStart != end)
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
        node.aabbMin = aabbUnion.Min();
        node.aabbMax = aabbUnion.Max();

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

        // If there is bad partition (location is start or end) then atleast put single node to a split
        if(partitionSplit == 0) partitionSplit += 1;
        else if(partitionSplit == static_cast<uint32_t>(end - start)) partitionSplit -= 1;        
        // Split Loc
        splitLoc = partitionSplit + start;
        
        // Init Nodes
        node.aabbMin = aabb.Min();
        node.aabbMax = aabb.Max();

        gpu.WaitMainStream();
    }
}

template <class PGroup>
SceneError GPUAccBVHGroup<PGroup>::InitializeGroup(// Accelerator Option Node
                                                   const SceneNodePtr& node,
                                                   // Map of hit keys for all materials
                                                   // w.r.t matId and primitive type
                                                   const std::map<TypeIdPair, HitKey>& allHitKeys,
                                                   // List of surface/material
                                                   // pairings that uses this accelerator type
                                                   // and primitive type
                                                   const std::map<uint32_t, IdPairs>& pairingList,
                                                   const std::vector<uint32_t>& transformList,
                                                   double time)
{

    const char* primGroupTypeName = primitiveGroup.Type();

    // Get params
    bool useStack = node->CommonBool(USE_STACK_NAME);
    params.useStack = useStack;

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
    // Allocate Empty Device memory Objects
    bvhDepths.resize(j, 0);
    bvhMemories.resize(j);
    bvhListMemory = std::move(DeviceMemory(j * sizeof(BVHNode<LeafData>*)));
    dBVHLists = static_cast<const BVHNode<LeafData>**>(bvhListMemory);

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
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);

    // Set Device
    const CudaGPU& gpu = system.BestGPU();//(*system.GPUList().begin());
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));

    uint32_t innerIndex = idLookup.at(surface);
    const PrimitiveRangeList& primRangeList = primitiveRanges[innerIndex];
    const PrimTransformType tType = primitiveGroup.TransformType();

    // Select Transform for construction
    const GPUTransformI* transform = dTransforms[dAccTransformIds[innerIndex]];
    if(tType == PrimTransformType::CONSTANT_LOCAL_TRANSFORM)
        transform = dTransforms[identityTransformIndex];

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
    size_t primIdsSize = totalPrimCount * sizeof(uint64_t);
    primIdsSize = Memory::AlignSize(primIdsSize);
    size_t idsSize = totalPrimCount * sizeof(uint32_t);
    idsSize = Memory::AlignSize(idsSize);
    size_t aabbsSize = totalPrimCount * sizeof(AABB3f);
    aabbsSize = Memory::AlignSize(aabbsSize);
    size_t primCentersSize = totalPrimCount * sizeof(Vector3f);
    primCentersSize = Memory::AlignSize(primCentersSize);
    size_t partitionOutSize = sizeof(uint32_t);
    partitionOutSize = Memory::AlignSize(partitionOutSize);
    //
    tempMemSize = Memory::AlignSize(tempMemSize);

    size_t totalSize = (primIdsSize +
                        idsSize * 2 +
                        aabbsSize +
                        primCentersSize +
                        partitionOutSize +
                        tempMemSize);


    DeviceMemory memory = DeviceMemory(totalSize);
    Byte* memPtr = static_cast<Byte*>(memory);
  
    // Memory Set
    size_t offset = 0;
    uint64_t* dPrimIds = reinterpret_cast<uint64_t*>(memPtr + offset);
    offset += primIdsSize;
    uint32_t* dIdsIn = reinterpret_cast<uint32_t*>(memPtr + offset);
    offset += idsSize;
    uint32_t* dIdsTemp = reinterpret_cast<uint32_t*>(memPtr + offset);
    offset += idsSize;
    AABB3f* dPrimAABBs = reinterpret_cast<AABB3f*>(memPtr + offset);
    offset += aabbsSize;
    Vector3f* dPrimCenters = reinterpret_cast<Vector3f*>(memPtr + offset);
    offset += primCentersSize;
    uint32_t* dPartitionSplitOut = reinterpret_cast<uint32_t*>(memPtr + offset);
    offset += partitionOutSize;
    void* dTemp = reinterpret_cast<void*>(memPtr + offset);
    offset += tempMemSize;
    assert(offset == totalSize);
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
                           KCInitIndices,
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
                       KCGenCenters<PGroup>,
                       //
                       dPrimCenters,
                       //
                       dIdsIn,
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
                        gpu,
                        // Call Related Args
                        current.parentId,
                        current.axis,
                        current.start, current.end);

        bvhNodes.emplace_back(node);
        uint32_t nextParentId = static_cast<uint32_t>(bvhNodes.size() - 1);
        SplitAxis nextSplit = DetermineNextSplit(current.axis, AABB3(node.aabbMin, node.aabbMax));

        // Update parent
        if(current.parentId != std::numeric_limits<uint32_t>::max())
        {
            if(current.left) bvhNodes[current.parentId].left = nextParentId;
            else bvhNodes[current.parentId].right = nextParentId;
        }

        // Check if not base case and add more generation
        if(splitLoc != std::numeric_limits<size_t>::max())
        {
            partitionQueue.emplace(SplitWork{true, current.start, splitLoc, nextSplit, nextParentId, current.depth + 1});
            partitionQueue.emplace(SplitWork{false, splitLoc, current.end, nextSplit, nextParentId, current.depth + 1});
            maxDepth = current.depth + 1;
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
    AABB3f accAABB(dBVHLists[innerIndex][0].aabbMin,
                   dBVHLists[innerIndex][0].aabbMax);
    surfaceAABBs.emplace(surface, accAABB);

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
    METU_LOG("Surface%u BVH(d=%u) generated in %f seconds.", 
             surface,
             maxDepth,
             t.Elapsed<CPUTimeSeconds>());

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
TracerError GPUAccBVHGroup<PGroup>::DestroyAccelerator(uint32_t surface,
                                                       const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccBVHGroup<PGroup>::DestroyAccelerators(const std::vector<uint32_t>& surfaces,
                                                        const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
size_t GPUAccBVHGroup<PGroup>::UsedGPUMemory() const
{
    size_t total = bvhListMemory.Size();
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
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
    
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
        dTransforms,
        dAccTransformIds,
        primitiveGroup.TransformType(),
        //
        primData
    );
}

template <class PGroup>
const SurfaceAABBList& GPUAccBVHGroup<PGroup>::AcceleratorAABBs() const
{
    return surfaceAABBs;
}