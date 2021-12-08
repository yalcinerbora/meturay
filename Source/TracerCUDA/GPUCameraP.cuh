#pragma once

#include "GPUCameraI.h"
#include "ImageStructs.h"
#include "GPUSurface.h"
#include "GPUPrimitiveEmpty.h"
#include "DeviceMemory.h"

template <class GPUCamera>
__global__ static
void KCCopyCamera(GPUCamera* newCamera,
                  const GPUCameraI* gRefCamera)
{
    if(threadIdx.x != 0) return;

    // We should safely up cast here
    const GPUCamera& gRefAsDerived = static_cast<const GPUCamera&>(*gRefCamera);
    // Use in-place new here
    new (newCamera) GPUCamera(gRefAsDerived);
}

template<class GPUCamera, class PGroup = GPUPrimitiveEmpty,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen = PGroup::GetSurfaceFunction>
class CPUCameraGroupP : public CPUCameraGroupI
{
    public:
        // Only use uv surface for now
        using Surface = UVSurface;
        // GPU Work Class will use this to specify the templated kernel
        using PrimitiveGroup        = PGroup;
        using GPUType               = GPUCamera;
        using HitData               = typename PGroup::HitData;
        using PrimitiveData         = typename PGroup::PrimitiveData;
        using SF                    = SurfaceFunc<Surface, HitData, PrimitiveData>;
        static constexpr SF         SurfF = SGen();

    private:
    protected:
        const PrimitiveGroup&           pg;

        DeviceMemory                    gpuCameraMemory;
        const GPUCamera*                dGPUCameras;

        // Temp CPU Allocations
        std::vector<uint16_t>           hMediumIds;
        std::vector<TransformId>        hTransformIds;
        std::vector<HitKey>             hWorkKeys;

        // Camera id to inner id
        std::map<uint32_t, uint32_t>    innerIds;

        GPUEndpointList				    gpuEndpointList;
        GPUCameraList				    gpuCameraList;
        uint32_t                        cameraCount;

        // Partially Implemented Interface
        SceneError				        InitializeCommon(const EndpointGroupDataList& lightNodes,
                                                         const TextureNodeMap& textures,
                                                         const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                         const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                         uint32_t batchId, double time,
                                                         const std::string& scenePath);
        size_t					        UsedCPUMemory() const override;
        size_t					        UsedGPUMemory() const override;
        // Helper
        void                            SetCameraLists();

    public:
        // Constructors & Destructor
                                        CPUCameraGroupP(const GPUPrimitiveGroupI& pg);
                                        ~CPUCameraGroupP() = default;


        // Fully Implemented Interface
        const GPUEndpointList&          GPUEndpoints() const override;
        const GPUCameraList&            GPUCameras() const override;
        uint32_t                        EndpointCount() const override;

        void                            CopyCamera(DeviceMemory&,
                                                   const GPUCameraI*,
                                                   const CudaSystem&) override;

        const std::vector<HitKey>&      PackedHitKeys() const override;
        uint32_t                        MaxInnerId() const override;

        const GPUCamera*                GPUCamerasDerived() const;
        const PrimitiveGroup&           PrimGroup() const;
};

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline CPUCameraGroupP<GPUCamera, PGroup, SGen>::CPUCameraGroupP(const GPUPrimitiveGroupI& eg)

    : pg(static_cast<const PGroup&>(eg))
    , dGPUCameras(nullptr)
    , cameraCount(0)
{}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
SceneError CPUCameraGroupP<GPUCamera, PGroup, SGen>::InitializeCommon(const EndpointGroupDataList& cameraNodes,
                                                                      const TextureNodeMap&,
                                                                      const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                                      const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                                      uint32_t batchId, double,
                                                                      const std::string&)
{
    SceneError err = SceneError::OK;

    cameraCount = static_cast<uint32_t>(cameraNodes.size());
    hMediumIds.reserve(cameraCount);
    hTransformIds.reserve(cameraCount);
    hWorkKeys.reserve(cameraCount);

    uint32_t innerIndex = 0;
    for(const auto& node : cameraNodes)
    {
        // Convert Ids to inner index
        uint16_t mediumIndex = static_cast<uint16_t>(mediumIdIndexPairs.at(node.mediumId));
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);

        hMediumIds.push_back(mediumIndex);
        hTransformIds.push_back(transformIndex);
        hWorkKeys.push_back(HitKey::CombinedKey(batchId, innerIndex));
        innerIds.emplace(node.endpointId, innerIndex);

        innerIndex++;
        if(innerIndex >= (1 << HitKey::IdBits))
            return SceneError::TOO_MANY_MATERIAL_IN_GROUP;
    }
    // Allocate data for texture references etc...
    GPUMemFuncs::AllocateMultiData(std::tie(dGPUCameras),
                                   gpuCameraMemory,
                                   {cameraCount});
    return SceneError::OK;
}


template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
void CPUCameraGroupP<GPUCamera, PGroup, SGen>::SetCameraLists()
{
    // Generate transform list
    for(uint32_t i = 0; i < cameraCount; i++)
    {
        const auto* ptr = static_cast<const GPUCameraI*>(dGPUCameras + i);
        gpuCameraList.push_back(ptr);
        gpuEndpointList.push_back(ptr);
    }
}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline const GPUEndpointList& CPUCameraGroupP<GPUCamera, PGroup, SGen>::GPUEndpoints() const
{
    return gpuEndpointList;
}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline const GPUCameraList& CPUCameraGroupP<GPUCamera, PGroup, SGen>::GPUCameras() const
{
    return gpuCameraList;
}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
const GPUCamera* CPUCameraGroupP<GPUCamera, PGroup, SGen>::GPUCamerasDerived() const
{
    return dGPUCameras;
}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline uint32_t CPUCameraGroupP<GPUCamera, PGroup, SGen>::EndpointCount() const
{
    return cameraCount;
}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
void CPUCameraGroupP<GPUCamera, PGroup, SGen>::CopyCamera(DeviceMemory& camMem,
                                                          const GPUCameraI* gCamera,
                                                          const CudaSystem& cudaSystem)
{
    GPUMemFuncs::EnlargeBuffer(camMem, sizeof(GPUCamera));
    CUDA_CHECK(cudaMemset(camMem, 0x00, sizeof(GPUCamera)));

    const auto& gpu = cudaSystem.BestGPU();
    gpu.KC_X(0, (cudaStream_t)0, 1,
             //
             KCCopyCamera<GPUCamera>,
             //
             static_cast<GPUCamera*>(camMem),
             gCamera);
    gpu.WaitMainStream();
}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
const std::vector<HitKey>& CPUCameraGroupP<GPUCamera, PGroup, SGen>::PackedHitKeys() const
{
    return hWorkKeys;
}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
uint32_t CPUCameraGroupP<GPUCamera, PGroup, SGen>::MaxInnerId() const
{
    return cameraCount;
}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline size_t CPUCameraGroupP<GPUCamera, PGroup, SGen>::UsedGPUMemory() const
{
    return gpuCameraMemory.Size();
}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline size_t CPUCameraGroupP<GPUCamera, PGroup, SGen>::UsedCPUMemory() const
{
    size_t totalSize = (hMediumIds.size() * sizeof(uint16_t) +
                        hTransformIds.size() * sizeof(TransformId) +
                        hWorkKeys.size() * sizeof(HitKey));

    return totalSize;
}

template<class GPUCamera, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline const PGroup& CPUCameraGroupP<GPUCamera, PGroup, SGen>::PrimGroup() const
{
    return pg;
}
