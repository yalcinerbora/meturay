#pragma once

#include "GPULightI.h"

#include "DeviceMemory.h"

#include "Texture.cuh"
#include "TextureReference.cuh"
#include "TextureReferenceGenerators.cuh"

using Tex2DMap = std::map<uint32_t, std::unique_ptr<TextureI<2>>>;

class GPULightP : public GPULightI
{
    protected:
        const TextureRefI<2, Vector3f>&    gRadianceRef;

    public:
        // Constructors & Destructor
        __device__              GPULightP(const TextureRefI<2, Vector3f>& gRadiance,
                                          // Base Class Related
                                          uint16_t mediumIndex,
                                          const GPUTransformI& gTransform);
        virtual                 ~GPULightP() = default;

        __device__ Vector3f     Emittance(const Vector3& wo,
                                          const Vector3& pos,
                                          //
                                          const Vector3f& normal,
                                          const Vector2f& uv) override;
        
};

template <class GPULight>
class CPULightGroupP : public CPULightGroupI
{
    public: 
        static constexpr const char*    RADIANCE_NAME = "radiance";

        using ConstructionInfo = TextureOrConstReferenceData<Vector3>;       

    private:
    protected:
        const CudaGPU&                  gpu;

        DeviceMemory                    gpuMemory;
        const GPULight*                 dGPULights;

        // Actual Texture Allocations
        Tex2DMap                            dTextureMemory;
        // Texture References (or Constant Reference)
        const ConstantRef<2, Vector3>*      dConstantRadiance;
        const TextureRef<2, Vector3>*       dTextureRadiance;
        const TextureRefI<2, Vector3>**     dRadiances;
        // Temp CPU Allocations
        std::vector<uint16_t>               hMediumIds;
        std::vector<TransformId>            hTransformIds;
        std::vector<ConstructionInfo>       hRadianceConstructionData;

        GPUEndpointList				        gpuLightList;
        uint32_t                            lightCount;

        // Partially Implemented Interface
        SceneError				        InitializeGroup(const EndpointGroupDataList& lightNodes,
                                                        const TextureNodeMap& textures,
                                                        const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                        const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                        uint32_t batchId, double time,
                                                        const std::string& scenePath) override;        
        size_t					        UsedCPUMemory() const override;

        // Helper
        void                            SetLightList();
        void                            ConstructTextureReferences();

    public:
        // Constructors & Destructor
                                        CPULightGroupP(const CudaGPU&);
                                        ~CPULightGroupP() = default;


        // Fully Implemented Interface
        const GPUEndpointList&          GPUEndpoints() const override;
        uint32_t                        EndpointCount() const override;
        size_t					        UsedGPUMemory() const override;
};

__device__
inline GPULightP::GPULightP(const TextureRefI<2, Vector3f>& gRadiance,
                            // Base Class Related
                            uint16_t mediumIndex,
                            const GPUTransformI& gTransform)
    : GPULightI(mediumIndex, gTransform)
    , gRadianceRef(gRadiance)
{}

__device__
inline Vector3f GPULightP::Emittance(const Vector3& wo,
                                     const Vector3& pos,
                                     //
                                     const Vector3f& normal,
                                     const Vector2f& uv)
{
    return gRadianceRef(uv);
}

template <class L>
inline CPULightGroupP<L>::CPULightGroupP(const CudaGPU& gpu)
    : gpu(gpu)
    , lightCount(0)
    , dGPULights(nullptr)
{}

template <class L>
SceneError CPULightGroupP<L>::InitializeGroup(const EndpointGroupDataList& lightNodes,
                                              const TextureNodeMap& textures,
                                              const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                              const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                              uint32_t batchId, double time,
                                              const std::string& scenePath)
{
    hMediumIds.reserve(lightCount);
    hTransformIds.reserve(lightCount);

    uint32_t innerIndex = 0;
    for(const auto& node : lightNodes)
    {
        // Convert Ids to inner index
        uint16_t mediumIndex = static_cast<uint16_t>(mediumIdIndexPairs.at(node.mediumId));
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);

        hMediumIds.push_back(mediumIndex);
        hTransformIds.push_back(transformIndex);

        innerIndex++;
    }

    size_t totalClassSize = sizeof(L) * lightCount;
    totalClassSize = Memory::AlignSize(totalClassSize);
    DeviceMemory::EnlargeBuffer(gpuMemory, totalClassSize);

    dGPULights = static_cast<const L*>(gpuMemory);

    return SceneError::OK;
}

template <class L>
void CPULightGroupP<L>::ConstructTextureReferences()
{
    const size_t counterCount = 2;
    DeviceMemory tempMemory;

    uint32_t* dCounters;
    ConstructionInfo* dRadianceConstructionData;
    DeviceMemory::AllocateMultiData(std::tie(dRadianceConstructionData, dCounters),
                                    tempMemory, {lightCount, counterCount});

    // Copy to GPU
    CUDA_CHECK(cudaMemset(dCounters, 0x00, sizeof(uint32_t) * counterCount));
    CUDA_CHECK(cudaMemcpy(dRadianceConstructionData,
                          hRadianceConstructionData.data(),
                          sizeof(ConstructionInfo) * materialCount,
                          cudaMemcpyHostToDevice));

    gpu.GridStrideKC_X(0, (cudaStream_t)0, lightCount,
                       GenerateEitherTexOrConstantReference<2, Vector3>,
                       const_cast<TextureRefI<2, Vector3f>**>(dRandiances),
                       const_cast<ConstantRef<2, Vector3>*>(dConstantRadiance),
                       const_cast<TextureRef<2, Vector3>*>(dTextureRadiance),
                       //
                       dCounters[0],
                       dCounters[1],
                       //
                       dRadianceConstructionData,
                       lightCount);
}


template <class L>
void CPULightGroupP<L>::SetLightList()
{
    // Generate transform list
    for(uint32_t i = 0; i < lightCount; i++)
    {
        const auto* ptr = static_cast<const GPULightI*>(dGPULights + i);
        gpuLightList.push_back(ptr);
    }
}

template <class L>
inline const GPUEndpointList& CPULightGroupP<L>::GPUEndpoints() const
{
    return gpuLightList;
}

template <class L>
inline uint32_t CPULightGroupP<L>::EndpointCount() const
{
    return lightCount;
}

template <class L>
inline size_t CPULightGroupP<L>::UsedGPUMemory() const
{
    return gpuMemory.Size();
}

template <class L>
inline size_t CPULightGroupP<L>::UsedCPUMemory() const
{
    size_t totalSize = (hMediumIds.size() * sizeof(uint16_t) +
                        hTransformIds.size() * sizeof(TransformId));

    return totalSize;
}