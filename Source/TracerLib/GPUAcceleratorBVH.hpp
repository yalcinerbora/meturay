
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
        primRangeList.fill(Zero2ul);
        hitKeyList.fill(HitKey::InvalidKey);

        const IdPairs& pList = pairings.second;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& p = pList[i];
            if(p.first == std::numeric_limits<uint32_t>::max()) break;

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
    assert(idLookup.size() == accRanges.size());

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
    using PrimitiveIndexStart = std::array<uint64_t, SceneConstants::MaxPrimitivePerSurface>;

    uint32_t index = idLookup.at(surface);
    const PrimitiveRangeList& primRangeList = primitiveRanges[index];
    
    size_t currentOffset = 0;
    PrimitiveRangeList indexOffsets;
    for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
    {
        const auto& p = primRangeList[i];
        //if(p.first == std::numeric_limits<uint32_t>::max()) break;

        indexOffsets[i][0] = currentOffset;
        currentOffset = p[1] - p[0];
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
                                        SpacePartition<PGroup>()));

    // CPU Memory
    std::vector<BVHNode<LeafData>> leafNodes;
    // GPU Memory
    DeviceMemory memory = DeviceMemory(totalPrimCount * sizeof(uint64_t));


    //uint64_t* dPrimIds = primIdList

    // Primitive Data is on GPU
    // we use pointers to partition an tempoary array;

        




    //void ConstBVHRecursive(BVHNode<LeafData> * &parentPtr,
    //                       SplitAxis axis,
    //                       size_t start, size_t end);

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