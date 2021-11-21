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

void GPUBaseAcceleratorLinear::GetReady(const CudaSystem&,
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
                                                       reinterpret_cast<void*>(KCIntersectBaseLinear));
    // Split work into multiple GPU's
    size_t offset = 0;
    int i = 0;
    for(const CudaGPU& gpu : system.SystemGPUs())
    {
        if(splits[i] == 0) break;
        // Generic
        const uint32_t workCount = static_cast<uint32_t>(splits[i]);
        gpu.GridStrideKC_X(0, (cudaStream_t)0,
                           workCount,
                           //
                           KCIntersectBaseLinear,
                           // Output
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

SceneError GPUBaseAcceleratorLinear::Initialize(// Accelerator Option Node
                                                const SceneNodePtr&,
                                                // List of surface to leaf accelerator ids
                                                const std::map<uint32_t, HitKey>& keyMap)
{
    idLookup.clear();

    leafCount = static_cast<uint32_t>(keyMap.size());
    size_t requiredSize = leafCount * sizeof(BaseLeaf);

    std::vector<HitKey> keyList(leafCount);
    uint32_t i = 0;
    for(const auto& pair : keyMap)
    {
        keyList[i] = pair.second;
        idLookup.emplace(pair.first, i);
        i++;
    }
    // Allocate and copy
    if(leafMemory.Size() < requiredSize)
        leafMemory = std::move(DeviceMemory(requiredSize));
    //
    dLeafs = static_cast<const BaseLeaf*>(leafMemory);
    Byte* keyLocation = static_cast<Byte*>(leafMemory) + offsetof(BaseLeaf, accKey);
    CUDA_CHECK(cudaMemcpy2D(keyLocation, sizeof(BaseLeaf),
                            keyList.data(), sizeof(HitKey),
                            sizeof(HitKey), leafCount,
                            cudaMemcpyHostToDevice));
    return SceneError::OK;
}

TracerError GPUBaseAcceleratorLinear::Construct(const CudaSystem&,
                                                // List of surface AABBs
                                                const SurfaceAABBList& aabbMap)
{
    if(aabbMap.size() != idLookup.size())
        return TracerError::UNABLE_TO_CONSTRUCT_BASE_ACCELERATOR;

    size_t aabbCount = static_cast<uint32_t>(aabbMap.size());
    std::vector<Vector3> aabbMin(aabbCount);
    std::vector<Vector3> aabbMax(aabbCount);

    sceneAABB = NegativeAABB3f;
    for(const auto& pair : aabbMap)
    {
        uint32_t index = idLookup.at(pair.first);
        aabbMin[index] = pair.second.Min();
        aabbMax[index] = pair.second.Max();
        sceneAABB.UnionSelf(AABB3f(pair.second.Min(),
                                   pair.second.Max()));
    }

    // Copy AABB data to leaf structs
    dLeafs = static_cast<const BaseLeaf*>(leafMemory);
    Byte* aabbMinLocation = static_cast<Byte*>(leafMemory) + offsetof(BaseLeaf, aabbMin);
    Byte* aabbMaxLocation = static_cast<Byte*>(leafMemory) + offsetof(BaseLeaf, aabbMax);
    CUDA_CHECK(cudaMemcpy2D(aabbMinLocation, sizeof(BaseLeaf),
                            aabbMin.data(), sizeof(Vector3f),
                            sizeof(Vector3f), aabbCount,
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(aabbMaxLocation, sizeof(BaseLeaf),
                            aabbMax.data(), sizeof(Vector3f),
                            sizeof(Vector3f), aabbCount,
                            cudaMemcpyHostToDevice));
    return TracerError::OK;
}

TracerError GPUBaseAcceleratorLinear::Destruct(const CudaSystem&)
{
    // Nothing to do here
    return TracerError::OK;
}

const AABB3f& GPUBaseAcceleratorLinear::SceneExtents() const
{
    return sceneAABB;
}

static_assert(IsTracerClass<GPUBaseAcceleratorLinear>::value,
              "GPUBaseAcceleratorLinear is not a Tracer Class.");

// Accelerator
template class GPUAccLinearGroup<GPUPrimitiveTriangle>;
template class GPUAccLinearGroup<GPUPrimitiveSphere>;

static_assert(IsTracerClass<GPUAccLinearGroup<GPUPrimitiveTriangle>>::value,
              "GPUAccLinearGroup<GPUPrimitiveTriangle> is not a Tracer Class.");
static_assert(IsTracerClass<GPUAccLinearGroup<GPUPrimitiveSphere>>::value,
              "GPUAccLinearGroup<GPUPrimitiveSphere> is not a Tracer Class.");