#pragma once

template <class PGroup>
GPUAccOptiXGroup<PGroup>::GPUAccOptiXGroup(const GPUPrimitiveGroupI& pGroup)
    : GPUAcceleratorGroup<PGroup>(pGroup)
    , optiXSystem(nullptr)
{}

template <class PGroup>
const char* GPUAccOptiXGroup<PGroup>::Type() const
{
    return TypeName();
}

template <class PGroup>
SceneError GPUAccOptiXGroup<PGroup>::InitializeGroup(// Accelerator Option Node
                                                     const SceneNodePtr& node,
                                                     // List of surface/material
                                                     // pairings that uses this accelerator type
                                                     // and primitive type
                                                     const std::map<uint32_t, SurfaceDefinition>& surfaceList,
                                                     double time)
{
    const char* primGroupTypeName = this->primitiveGroup.Type();

    std::vector<uint32_t> hTransformIndices;
    hTransformIndices.reserve(surfaceList.size());

    // Iterate over pairings
    int surfaceInnerId = 0;
    for(const auto& surface : surfaceList)
    {
        PrimitiveRangeList primRangeList;
        HitKeyList hitKeyList;
        primRangeList.fill(Vector2ul(std::numeric_limits<uint64_t>::max()));
        hitKeyList.fill(HitKey::InvalidKey);

        const IdKeyPairs& pList = surface.second.primIdWorkKeyPairs;
        for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
        {
            const IdKeyPair& p = pList[i];
            if(p.first == std::numeric_limits<uint32_t>::max())
                break;

            primRangeList[i] = this->primitiveGroup.PrimitiveBatchRange(p.first);
            hitKeyList[i] = p.second;
        }

        hTransformIndices.push_back(surface.second.globalTransformIndex);
        primitiveRanges.push_back(primRangeList);
        primitiveMaterialKeys.push_back(hitKeyList);
        idLookup.emplace(surface.first, surfaceInnerId);
        keyExpandOption.push_back(surface.second.doKeyExpansion);
        surfaceInnerId++;
    }
    assert(surfaceInnerId == hTransformIndices.size());

    // Reserve Device Memories for traversal
    optixTraverseMemory.resize(hTransformIndices.size());

    //DeviceMemory::EnlargeBuffer(memory,
    //                            sizeof(TransformId) * hTransformIndices.size());

    return SceneError::INTERNAL_DUPLICATE_ACCEL_ID;
}

template <class PGroup>
uint32_t GPUAccOptiXGroup<PGroup>::InnerId(uint32_t surfaceId) const
{
    return idLookup.at(surfaceId);
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::ConstructAccelerators(const CudaSystem& system)
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
TracerError GPUAccOptiXGroup<PGroup>::ConstructAccelerator(uint32_t surface,
                                                           const CudaSystem& system)
{
    return TracerError::UNABLE_TO_CONSTRUCT_ACCELERATOR;
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::ConstructAccelerators(const std::vector<uint32_t>& surfaces,
                                                            const CudaSystem& system)
{
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
TracerError GPUAccOptiXGroup<PGroup>::DestroyAccelerators(const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::DestroyAccelerator(uint32_t surface,
                                                         const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
TracerError GPUAccOptiXGroup<PGroup>::DestroyAccelerators(const std::vector<uint32_t>& surfaces,
                                                          const CudaSystem&)
{
    // TODO: Implement
    return TracerError::OK;
}

template <class PGroup>
size_t GPUAccOptiXGroup<PGroup>::UsedGPUMemory() const
{
    // TODO:
    return 0;
}

template <class PGroup>
size_t GPUAccOptiXGroup<PGroup>::UsedCPUMemory() const
{
    // TODO:
    // Write allocator wrapper for which keeps track of total bytes allocated
    // and deallocated
    return 0;
}

template <class PGroup>
void GPUAccOptiXGroup<PGroup>::Hit(const CudaGPU& gpu,
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
    // We do nothing on hit this function should not be called
    // Since base accelerator will handle entire hit operation
}

template <class PGroup>
const SurfaceAABBList& GPUAccOptiXGroup<PGroup>::AcceleratorAABBs() const
{
    return surfaceAABBs;
}

template <class PGroup>
void GPUAccOptiXGroup<PGroup>::SetOptiXSystem(const OptiXSystem* sys)
{
    optiXSystem = sys;
}