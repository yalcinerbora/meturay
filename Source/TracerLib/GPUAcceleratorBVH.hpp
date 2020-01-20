
#include "TracerDebug.h"

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
SplitAxis GPUAccBVHGroup<PGroup>::DetermineNextSplit(SplitAxis split, const AABB3f& aabb)
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

    ////METU_LOG("start end %zu %zu", start, end);
    //if(start == 81171)
    //    METU_LOG("!!!!!!!!start split end - %zu %zu", start, end);

    // Base Case (CPU Mode)
    if(end - start == 1)
    {
        PrimitiveId id = dPrimIds[start];        
        HitKey matKey = FindHitKey(accIndex, id);
        
        node.isLeaf = true;
        node.leaf = PGroup::LeafFunc(matKey, id, primData);
        splitLoc = std::numeric_limits<size_t>::max();
    }
    else if(end - start <= Threshold_CPU_GPU)
    {
        AABB3f aabbUnion = NegativeAABB3;
        float avgCenter = 0.0f;
        // Find AABB
        for(size_t i = start; i < end; i++)
        {
            uint32_t index = dIndicesIn[i];
            AABB3f aabb = dAABBs[index];
            float center = dPrimCenters[index][axisIndex];
            aabbUnion.UnionSelf(aabb);
            avgCenter = (avgCenter * (i - start) + center) / (i - start + 1);
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
                leftTriAxisCenter = dPrimCenters[index][axisIndex];
            }
            while(leftTriAxisCenter >= avgCenter);
            float rightTriAxisCenter;
            do
            {
                if(splitEnd <= static_cast<int64_t>(start + 1)) break;
                splitEnd--;
                uint32_t index = dIndicesIn[splitEnd];
                rightTriAxisCenter = dPrimCenters[index][axisIndex];
            }
            while(rightTriAxisCenter <= avgCenter);

            if(splitStart < splitEnd)
                std::swap(dIndicesIn[splitEnd], dIndicesIn[splitStart]);
        }
        node.aabbMin = aabbUnion.Min();
        node.aabbMax = aabbUnion.Max();
       
        //METU_LOG("start split end - %zu %zu %zu", start, splitStart, end);
        //if(splitStart == 81171)
        //    METU_LOG("!!!!!!!!start split end - %zu %zu %zu", start, splitStart, end);
        splitLoc = splitStart;

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
            //Debug::DumpMemToFile("aabbs", aabbTemp0, reductionCount);
            

        } while(reductionCount != 1);

        // Copy Host
        CUDA_CHECK(cudaMemcpy(&aabb, aabbTemp1, sizeof(AABB3f), cudaMemcpyDeviceToHost));

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
        splitLoc = partitionSplit + start;

        // Init Nodes
        node.aabbMin = aabb.Min();
        node.aabbMax = aabb.Max();

        gpu.WaitMainStream();
    }
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

template<class T>
std::ostream& operator<<(std::ostream& stream, const BVHNode<T>& v)
{
    stream << std::setw(0);
    if(v.isLeaf)
    {
        stream << "leaf";
    }
    else
    {
        stream << "[ " << v.left << ", " << v.right << ", " << v.parent << " ]";
        stream << AABB3(v.aabbMin, v.aabbMax);
    }
    return stream;
}

template <class PGroup>
void GPUAccBVHGroup<PGroup>::ConstructAccelerator(uint32_t surface,
                                                  const CudaSystem& system)
{        
    using PrimitiveData = typename PGroup::PrimitiveData;
    using PrimitiveIndexStart = std::array<uint64_t, SceneConstants::MaxPrimitivePerSurface>;
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
    

    // Set Device
    const CudaGPU& gpu = (*system.GPUList().begin());
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));

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
        currentOffset += (range[1] - range[0]);
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
    size_t totalSize = totalPrimCount * (sizeof(uint64_t) +
                                         2 * sizeof(uint32_t) +
                                         sizeof(AABB3f) +
                                         sizeof(Vector3f)) +
                       tempMemSize + 
                       sizeof(uint32_t);
    DeviceMemory memory = DeviceMemory(totalSize);
    Byte* memPtr = static_cast<Byte*>(memory);
    size_t offset = 0;
    Vector3f* dPrimCenters = reinterpret_cast<Vector3f*>(memPtr + offset);
    offset += totalPrimCount * sizeof(Vector3f);
    AABB3f* dPrimAABBs = reinterpret_cast<AABB3f*>(memPtr + offset);
    offset += totalPrimCount * sizeof(AABB3f);
    uint64_t* dPrimIds = reinterpret_cast<uint64_t*>(memPtr + offset);
    offset += totalPrimCount * sizeof(uint64_t);
    uint32_t* dIdsIn = reinterpret_cast<uint32_t*>(memPtr + offset);
    offset += totalPrimCount * sizeof(uint32_t);
    uint32_t* dIdsTemp = reinterpret_cast<uint32_t*>(memPtr + offset);
    offset += totalPrimCount * sizeof(uint32_t);
    void* dTemp = reinterpret_cast<void*>(memPtr + offset);
    offset += tempMemSize;
    uint32_t* dPartitionSplitOut = reinterpret_cast<uint32_t*>(memPtr + offset);
    offset += sizeof(uint32_t);
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
                       AABBGen<PGroup>(primData),
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
                        index,
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
    // Finally Nodes are Generated now copy it to GPU Memory
    bvhMemories[index] = std::move(DeviceMemory(sizeof(BVHNode<LeafData>) * bvhNodes.size()));
    bvhDepths[index] = maxDepth;

    Debug::DumpMemToFile("BVHNodes", bvhNodes.data(), bvhNodes.size());
    
    CUDA_CHECK(cudaMemcpy(bvhMemories[index],
                          bvhNodes.data(),
                          sizeof(BVHNode<LeafData>) * bvhNodes.size(),
                          cudaMemcpyHostToDevice));

    BVHNode<LeafData>* dBVHStart = static_cast<BVHNode<LeafData>*>(bvhMemories[index]);
    CUDA_CHECK(cudaMemcpy(dBVHLists + index,
                          &dBVHStart,
                          sizeof(BVHNode<LeafData>*),
                          cudaMemcpyHostToDevice));
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