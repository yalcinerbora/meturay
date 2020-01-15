
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