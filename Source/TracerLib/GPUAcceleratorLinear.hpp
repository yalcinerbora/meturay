
template <class PGroup>
GPUAccLinearGroup<PGroup>::GPUAccLinearGroup(const GPUPrimitiveGroupI& pGroup,
                                             const TransformStruct* dInvTransforms)
    : GPUAcceleratorGroup<PGroup>(pGroup, dInvTransforms)
    , dAccRanges(nullptr)
    , dLeafList(nullptr)
{}

template<class PGroup>
const char* LinearAccelTypeName<PGroup>::TypeName()
{
    static const std::string typeName = std::string("Linear") + PGroup::TypeName();
    return typeName.c_str();
}

template <class PGroup>
const char* GPUAccLinearGroup<PGroup>::Type() const
{
    return TypeName();
}

template <class PGroup>
SceneError GPUAccLinearGroup<PGroup>::InitializeGroup(// Map of hit keys for all materials
                                                      // w.r.t matId and primitive type
                                                      const std::map<TypeIdPair, HitKey>& allHitKeys,
                                                      // List of surface/material
                                                      // pairings that uses this accelerator type
                                                      // and primitive type
                                                      const std::map<uint32_t, IdPairs>& pairingList,
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
        PrimitiveRangeList primRangeList;
        HitKeyList hitKeyList;
        primRangeList.fill(Zero2ul);
        hitKeyList.fill(HitKey::InvalidKey);
        
        Vector2ul range = Vector2ul(totalSize, 0);

        size_t localSize = 0;
        const IdPairs& pList = pairings.second;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const auto& p = pList[i];
            if(p.first == std::numeric_limits<uint32_t>::max()) break;

            primRangeList[i] = primitiveGroup.PrimitiveBatchRange(p.second);
            hitKeyList[i] = allHitKeys.at(std::make_pair(primGroupTypeName, p.first));
            localSize += primRangeList[i][1] - primRangeList[i][0];
        }
        range[1] = range[0] + localSize;
        totalSize += localSize;
        
        // Put generated AABB
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
    size_t leafDataSize = totalSize * sizeof(LeafData);
    size_t accRangeSize = idLookup.size() * sizeof(Vector2ul);
    memory = std::move(DeviceMemory(leafDataSize + accRangeSize));
    dLeafList = static_cast<LeafData*>(memory);
    dAccRanges = reinterpret_cast<Vector2ul*>(static_cast<uint8_t*>(memory) + leafDataSize);

    // Copy Leaf counts to cpu memory
    CUDA_CHECK(cudaMemcpy(dAccRanges, accRanges.data(), accRangeSize,
                          cudaMemcpyHostToDevice));

    return SceneError::OK;
}

template <class PGroup>
SceneError GPUAccLinearGroup<PGroup>::ChangeTime(// Map of hit keys for all materials
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
uint32_t GPUAccLinearGroup<PGroup>::InnerId(uint32_t surfaceId) const
{
    return idLookup.at(surfaceId);
}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::ConstructAccelerators(const CudaSystem& system)
{
    // TODO: make this a single KC
    for(const auto& id : idLookup)
    {
        ConstructAccelerator(id.first, system);
    }
}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::ConstructAccelerator(uint32_t surface,
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
    const CudaGPU& gpu = *system.GPUList().begin();
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
}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::ConstructAccelerators(const std::vector<uint32_t>& surfaces,
                                                      const CudaSystem& system)
{
    // TODO: make this a single KC
    for(const uint32_t& id : surfaces)
    {
        ConstructAccelerator(id, system);
    }
}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::DestroyAccelerators(const CudaSystem&)
{
    //...
    // Define destory??
    // There is no destruction or deallocation
}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::DestroyAccelerator(uint32_t surface,
                                                   const CudaSystem&)
{
    //...
    // Define destory??
    // There is no destruction or deallocation
}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::DestroyAccelerators(const std::vector<uint32_t>& surfaces,
                                                    const CudaSystem&)
{
    //...
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
GPUAccLinearBatch<PGroup>::GPUAccLinearBatch(const GPUAcceleratorGroupI& a,
                                             const GPUPrimitiveGroupI& p)
    : GPUAcceleratorBatch(a, p)
{}

template <class PGroup>
const char* GPUAccLinearBatch<PGroup>::Type() const
{
    return TypeName();
}

template <class PGroup>
void GPUAccLinearBatch<PGroup>::Hit(const CudaGPU& gpu,
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
        KCIntersectLinear<PGroup>,
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
        acceleratorGroup.dLeafList,
        acceleratorGroup.dAccRanges,
        acceleratorGroup.dInverseTransforms,
        //
        primData
    );
}