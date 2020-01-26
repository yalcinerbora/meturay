#include "GPUAcceleratorLinear.cuh"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

#include "RayLib/ObjectFuncDefinitions.h"

const char* GPUBaseAcceleratorLinear::TypeName()
{
    return "Linear";
}

GPUBaseAcceleratorLinear::GPUBaseAcceleratorLinear()
    : dLeafs(nullptr)
    , dPrevLocList(nullptr)
{}

const char* GPUBaseAcceleratorLinear::Type() const
{
    return TypeName();
}

void GPUBaseAcceleratorLinear::GetReady(const CudaSystem& system, 
                                        uint32_t rayCount)
{
    size_t requiredSize = rayCount * sizeof(uint32_t);
    if(rayLocMemory.Size() < requiredSize)
        rayLocMemory = std::move(DeviceMemory(requiredSize));        

    dPrevLocList = static_cast<uint32_t*>(rayLocMemory);
    CUDA_CHECK(cudaMemset(dPrevLocList, 0x00, requiredSize));
}

void GPUBaseAcceleratorLinear::Hit(const CudaSystem& system,
                                   // Output
                                   TransformId* dTransformIds,
                                   HitKey* dAcceleratorKeys,
                                   // Inputs
                                   const RayGMem* dRays,
                                   const RayId* dRayIds,
                                   const uint32_t rayCount) const
{
    // Split work
    const auto splits = system.GridStrideMultiGPUSplit(rayCount,
                                                       StaticThreadPerBlock1D,
                                                       0,
                                                       KCIntersectBaseLinear);
    // Split work into multiple GPU's
    size_t offset = 0;
    int i = 0;
    for(const CudaGPU& gpu : system.GPUList())
    {
        if(splits[i] == 0) break;
        // Generic
        const uint32_t workCount = static_cast<uint32_t>(splits[i]);
        gpu.GridStrideKC_X(0, (cudaStream_t)0,
                           workCount,
                           //
                           KCIntersectBaseLinear,
                           // Output
                           dTransformIds,
                           dAcceleratorKeys + offset,
                           // I-O
                           dPrevLocList,
                           // Input
                           dRays,
                           dRayIds + offset,
                           workCount,
                           // Constants
                           dLeafs,
                           leafCount);
        i++;
        offset += workCount;
    }
}

SceneError GPUBaseAcceleratorLinear::Initialize(// List of surface to transform id hit key mappings
                                                const std::map<uint32_t, BaseLeaf>& map)
{
    innerIds.clear();

    leafCount = static_cast<uint32_t>(map.size());
    size_t requiredSize = leafCount * sizeof(BaseLeaf);

    std::vector<BaseLeaf> leafCPU(leafCount);
    uint32_t i = 0;
    for(const auto& pair : map)
    {
        leafCPU[i] = pair.second;
        innerIds.emplace(pair.first, i);
        i++;
    }

    // Allocate and copy
    if(leafMemory.Size() < requiredSize)
        leafMemory = std::move(DeviceMemory(requiredSize));

    dLeafs = static_cast<const BaseLeaf*>(leafMemory);
    CUDA_CHECK(cudaMemcpy(const_cast<BaseLeaf*>(dLeafs),
                          leafCPU.data(), sizeof(BaseLeaf) * leafCount,
                          cudaMemcpyHostToDevice));
    return SceneError::OK;
}

SceneError GPUBaseAcceleratorLinear::Change(// List of only changed surface to transform id hit key mappings
                                            const std::map<uint32_t, BaseLeaf>& map)
{
    for(const auto& pair : map)
    {
        // Use managed memory functionality
        uint32_t index = innerIds[pair.first];
        const_cast<BaseLeaf*>(dLeafs)[index] = pair.second;
    }
    return SceneError::OK;
}

TracerError GPUBaseAcceleratorLinear::Constrcut(const CudaSystem&)
{
    // Nothing to do here
    return TracerError::OK;
}

TracerError GPUBaseAcceleratorLinear::Destruct(const CudaSystem&)
{
    // Nothing to do here
    return TracerError::OK;
}

static_assert(IsTracerClass<GPUBaseAcceleratorLinear>::value,
              "GPUBaseAcceleratorLinear is not a Tracer Class.");

// Accelerator
template class GPUAccLinearGroup<GPUPrimitiveTriangle>;
template class GPUAccLinearGroup<GPUPrimitiveSphere>;

template class GPUAccLinearBatch<GPUPrimitiveTriangle>;
template class GPUAccLinearBatch<GPUPrimitiveSphere>;

static_assert(IsTracerClass<GPUAccLinearGroup<GPUPrimitiveTriangle>>::value,
              "GPUAccLinearGroup<GPUPrimitiveTriangle> is not a Tracer Class.");
static_assert(IsTracerClass<GPUAccLinearGroup<GPUPrimitiveSphere>>::value,
              "GPUAccLinearGroup<GPUPrimitiveSphere> is not a Tracer Class.");