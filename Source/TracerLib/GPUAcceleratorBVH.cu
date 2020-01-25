#include "GPUAcceleratorBVH.cuh"
#include "TypeTraits.h"

#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

const char* GPUBaseAcceleratorBVH::TypeName()
{
    return "BasicBVH";
}

GPUBaseAcceleratorBVH::GPUBaseAcceleratorBVH()
    : dBVH(nullptr)
    , dRayStates(nullptr)
{}

const char* GPUBaseAcceleratorBVH::Type() const
{
    return TypeName();
}

void GPUBaseAcceleratorBVH::GetReady(uint32_t rayCount)
{

}

void GPUBaseAcceleratorBVH::Hit(const CudaSystem&,
                                // Output
                                TransformId* dTransformIds,
                                HitKey* dAcceleratorKeys,
                                // Inputs
                                const RayGMem* dRays,
                                const RayId* dRayIds,
                                const uint32_t rayCount) const
{

}

SceneError GPUBaseAcceleratorBVH::Initialize(// List of surface to transform id hit key mappings
                                             const std::map<uint32_t, BaseLeaf>&)
{
    return SceneError::OK;
}

SceneError GPUBaseAcceleratorBVH::Change(// List of only changed surface to transform id hit key mappings
                                         const std::map<uint32_t, BaseLeaf>&)
{
    return SceneError::OK;
}

TracerError GPUBaseAcceleratorBVH::Constrcut(const CudaSystem&)
{
    return TracerError::OK;
}

TracerError GPUBaseAcceleratorBVH::Destruct(const CudaSystem&)
{
    return TracerError::OK;
}

// Accelerator Instancing for basic primitives
template class GPUAccBVHGroup<GPUPrimitiveTriangle>;
template class GPUAccBVHGroup<GPUPrimitiveSphere>;

template class GPUAccBVHBatch<GPUPrimitiveTriangle>;
template class GPUAccBVHBatch<GPUPrimitiveSphere>;

static_assert(IsTracerClass<GPUAccBVHGroup<GPUPrimitiveTriangle>>::value,
              "GPUAccBVHBatch<GPUPrimitiveTriangle> is not a Tracer Class.");
static_assert(IsTracerClass<GPUAccBVHGroup<GPUPrimitiveSphere>>::value,
              "GPUAccBVHGroup<GPUPrimitiveSphere> is not a Tracer Class.");