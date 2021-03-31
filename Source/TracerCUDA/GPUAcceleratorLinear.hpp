template <class PGroup>
GPUAccLinearGroup<PGroup>::GPUAccLinearGroup(const GPUPrimitiveGroupI& pGroup)
    : GPUAcceleratorGroup<PGroup>(pGroup)
    , dAccRanges(nullptr)
    , dLeafList(nullptr)
{}

template <class PGroup>
const char* GPUAccLinearGroup<PGroup>::Type() const
{
    return TypeName();
}

template <class PGroup>
SceneError GPUAccLinearGroup<PGroup>::InitializeGroup(// Accelerator Option Node
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
    accRanges.clear();
    primitiveRanges.clear();
    primitiveMaterialKeys.clear();
    idLookup.clear();

    const char* primGroupTypeName = primitiveGroup.Type();

    // Iterate over pairings
    int j = 0;
    size_t totalSize = 0;
    for(const auto& pairings : pairingList)
    {
        PrimitiveIdList primIdList;
        PrimitiveRangeList primRangeList;
        HitKeyList hitKeyList;
        primRangeList.fill(Zero2ul);
        primIdList.fill(std::numeric_limits<uint32_t>::max());
        hitKeyList.fill(HitKey::InvalidKey);

        Vector2ul range = Vector2ul(totalSize, 0);

        size_t localSize = 0;
        const IdPairs& pList = pairings.second;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& p = pList[i];
            if(p.first == std::numeric_limits<uint32_t>::max()) break;

            primIdList[i] = p.second;
            primRangeList[i] = primitiveGroup.PrimitiveBatchRange(p.second);
            hitKeyList[i] = allHitKeys.at(std::make_pair(primGroupTypeName, p.first));
            localSize += primRangeList[i][1] - primRangeList[i][0];
        }
        range[1] = range[0] + localSize;
        totalSize += localSize;

        // Put generated AABB
        primitiveIds.push_back(primIdList);
        primitiveRanges.push_back(primRangeList);
        primitiveMaterialKeys.push_back(hitKeyList);
        accRanges.push_back(range);
        idLookup.emplace(pairings.first, j);
        j++;
    }
    assert(primitiveRanges.size() == primitiveMaterialKeys.size());
    assert(primitiveMaterialKeys.size() == idLookup.size());
    assert(idLookup.size() == accRanges.size());

    // Allocate memory
    assert(j == transformList.size());
    uint32_t bvhCount = j;

    // Finally Allocate and load to GPU memory
    size_t sizeOfTransformIndices = sizeof(uint32_t) * bvhCount;
    sizeOfTransformIndices = Memory::AlignSize(sizeOfTransformIndices);
    size_t leafDataSize = totalSize * sizeof(LeafData);
    leafDataSize = Memory::AlignSize(leafDataSize);
    size_t accRangeSize = idLookup.size() * sizeof(Vector2ul);
    accRangeSize = Memory::AlignSize(accRangeSize);

    size_t requiredSize = (sizeOfTransformIndices + leafDataSize + accRangeSize);

    // Reallocate if memory is not enough
    DeviceMemory::EnlargeBuffer(memory, requiredSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(memory);
    dAccTransformIds = reinterpret_cast<uint32_t*>(dBasePtr + offset);
    offset += sizeOfTransformIndices;
    dLeafList = reinterpret_cast<LeafData*>(dBasePtr + offset);
    offset += leafDataSize;
    dAccRanges = reinterpret_cast<Vector2ul*>(dBasePtr + offset);
    offset += accRangeSize;
    assert(requiredSize == offset);

    // Copy Leaf counts to cpu memory
    CUDA_CHECK(cudaMemcpy(dAccRanges, accRanges.data(),
                          idLookup.size() * sizeof(Vector2ul),
                          cudaMemcpyHostToDevice));
        // Copy Transforms
    CUDA_CHECK(cudaMemcpy(dAccTransformIds,
                          transformList.data(),
                          bvhCount * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    return SceneError::OK;
}

template <class PGroup>
uint32_t GPUAccLinearGroup<PGroup>::InnerId(uint32_t surfaceId) const
{
    return idLookup.at(surfaceId);
}

template <class PGroup>
TracerError GPUAccLinearGroup<PGroup>::ConstructAccelerators(const CudaSystem& system)
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
TracerError GPUAccLinearGroup<PGroup>::ConstructAccelerator(uint32_t surface,
                                                            const CudaSystem& system)
{
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);

    const uint32_t index = idLookup.at(surface);
    //const Vector2ul& accelRange = acceleratorRanges[index];
    const PrimitiveRangeList& rangeList = primitiveRanges[index];
    const HitKeyList& hitList = primitiveMaterialKeys[index];

    // Copy Locally to stack to send it to const memory of the GPU
    HKList hkList = {{}};
    PRList prList = {{}};
    std::memcpy(const_cast<HitKey*>(hkList.materialKeys), hitList.data(),
                sizeof(HitKey) * SceneConstants::MaxPrimitivePerSurface);
    std::memcpy(const_cast<Vector2ul*>(prList.primRanges), rangeList.data(),
                sizeof(Vector2ul) * SceneConstants::MaxPrimitivePerSurface);

    size_t workCount = accRanges[index][1] - accRanges[index][0];

    // TODO: Select a GPU
    const CudaGPU& gpu = system.BestGPU();//*system.GPUList().begin();
    // KC
    gpu.AsyncGridStrideKC_X
    (
        0,
        workCount,
        //
        KCConstructLinear<PGroup>,
        // Args
        // O
        dLeafList,
        // Input
        dAccRanges,
        hkList,
        prList,
        primData,
        index
    );

    // Acquire GPU Transform Ptr for this accelerator to the CPU
    const GPUTransformI* transform = nullptr;
    AcquireAcceleratorGPUTransform(transform,
                                   dAccTransformIds,
                                   dTransforms,
                                   index,
                                   gpu);

    // Check if this accelerator utilizes constant transform
    if(primitiveGroup.TransformType() == PrimTransformType::CONSTANT_LOCAL_TRANSFORM)
    {
        // Rays are going to be transformed
        // utilize local space primitive AABB
        AABB3f unionAABB = NegativeAABB3f;
        for(uint32_t primBatchId : primitiveIds[index])
        {
            if(primBatchId == std::numeric_limits<uint32_t>::max()) break;

            AABB3f aabb = primitiveGroup.PrimitiveBatchAABB(primBatchId);
            unionAABB.UnionSelf(aabb);
        }

        // Transform this AABB to world space
        // since base Accelerator works on world space
        TransformLocalAABBToWorld(unionAABB, *transform, gpu);

        surfaceAABBs.emplace(surface, unionAABB);
    }
    else
    {
        // Use transformation & primitive data to generate AABB
        size_t totalAABBCount = 0;
        for(const auto& primRange : primitiveRanges[index])
        {
            if(primRange == Zero2ul) break;
            totalAABBCount = primRange[0] - primRange[1];
        }

        // Check cub reduction temp storage size
        size_t cubTempStorageSize = 0;
        AABB3f* dInAABBs = nullptr;
        AABB3f* dOutAABB = nullptr;
        CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr, cubTempStorageSize,
                                             dInAABBs, dOutAABB,
                                             static_cast<int>(totalAABBCount),
                                             AABBUnion(), NegativeAABB3f));

        size_t aabbSizes = (totalAABBCount + 1) * sizeof(AABB3f);
        DeviceMemory memory(aabbSizes + cubTempStorageSize);

        size_t offset = 0;
        Byte* memPtr = static_cast<Byte*>(memory);
        dInAABBs = reinterpret_cast<AABB3f*>(memPtr + offset);
        offset += totalAABBCount * sizeof(AABB3f);
        dOutAABB = reinterpret_cast<AABB3f*>(memPtr + offset);
        offset += sizeof(AABB3f);
        void* dTempStorage = memPtr + offset;
        offset += cubTempStorageSize;
        assert(offset == memory.Size());

        CUDA_CHECK(cub::DeviceReduce::Reduce(dTempStorage, cubTempStorageSize,
                                             dInAABBs, dOutAABB,
                                             static_cast<int>(totalAABBCount),
                                             AABBUnion(), NegativeAABB3f));

        // First Generate AABBs from Primitives
        // TODO:
        uint32_t aabbOffset = 0;
        for(uint32_t i = 0; i < primitiveRanges[index].size(); i++)
        {
            Vector2ul primRange = primitiveRanges[index][i];
            uint32_t aabbCount = static_cast<uint32_t>(primRange[1] - primRange[0]);

            gpu.GridStrideKC_X(0, 0,
                               aabbCount,
                               //
                               KCGenAABBs<PGroup>,
                               //
                               dInAABBs + aabbOffset,
                               //
                               primRange,
                               //
                               AABBGen<PGroup>(primData, *transform),
                               aabbCount);

            aabbOffset += aabbCount;
        }
        assert(aabbOffset = static_cast<uint32_t>(accRanges[index][1] - accRanges[index][0]));

        // Sync device to access gpu memory from host
        AABB3f hostAABB;
        CUDA_CHECK(cudaMemcpy(&hostAABB, dOutAABB, sizeof(AABB3f),
                              cudaMemcpyDeviceToHost));

        surfaceAABBs.emplace(surface, hostAABB);
    }

    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccLinearGroup<PGroup>::ConstructAccelerators(const std::vector<uint32_t>& surfaces,
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
TracerError GPUAccLinearGroup<PGroup>::DestroyAccelerators(const CudaSystem&)
{
    //...
    // Define destory??
    // There is no destruction or deallocation
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccLinearGroup<PGroup>::DestroyAccelerator(uint32_t surface,
                                                          const CudaSystem&)
{
    //...
    // Define destory??
    // There is no destruction or deallocation
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccLinearGroup<PGroup>::DestroyAccelerators(const std::vector<uint32_t>& surfaces,
                                                           const CudaSystem&)
{
    //...
    // Define destory??
    // There is no destruction or deallocation
    return TracerError::OK;
}

template <class PGroup>
size_t GPUAccLinearGroup<PGroup>::UsedGPUMemory() const
{
    return memory.Size();
}

template <class PGroup>
size_t GPUAccLinearGroup<PGroup>::UsedCPUMemory() const
{
    // TODO:
    // Write allocator wrapper for which keeps track of total bytes allocated
    // and deallocated
    return 0;
}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::Hit(const CudaGPU& gpu,
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
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);

    gpu.AsyncGridStrideKC_X
    (
        0,
        rayCount,
        //
        KCIntersectLinear<PGroup>,
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
        dLeafList,
        dAccRanges,
        dTransforms,
        dAccTransformIds,
        primitiveGroup.TransformType(),
        //
        primData
    );
}

template <class PGroup>
const SurfaceAABBList& GPUAccLinearGroup<PGroup>::AcceleratorAABBs() const
{
    return surfaceAABBs;
}