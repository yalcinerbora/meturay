#pragma once

#include "GPULightI.h"

#include "DeviceMemory.h"

#include "Texture.cuh"
#include "TextureReference.cuh"
#include "TextureReferenceGenerators.cuh"
#include "TextureFunctions.h"
#include "GPUSurface.h"

using Tex2DMap = std::map<uint32_t, std::unique_ptr<TextureI<2>>>;

// CUDA complains when generator function
// is called as a static member function
// instead we supply it as a template parameter
template <class S, class H, class D>
using SurfaceFuncGenerator = SurfaceFunc<S, H, D>(*)();

class GPULightP : public GPULightI
{
    protected:
        const TextureRefI<2, Vector3f>&    gRadianceRef;

    public:
        // Constructors & Destructor
        __device__              GPULightP(const TextureRefI<2, Vector3f>& gRadiance,
                                          // Base Class Related
                                          uint16_t mediumIndex, HitKey,
                                          const GPUTransformI& gTransform);
        virtual                 ~GPULightP() = default;

        __device__ Vector3f     Emit(const Vector3& wo,
                                     const Vector3& pos,
                                     //
                                     const UVSurface&) const override;

};

template<class GPULight, class PGroup = GPUPrimitiveEmpty,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen = PGroup::GetSurfaceFunction>
class CPULightGroupP : public CPULightGroupI
{
    public:
        static constexpr const char*    RADIANCE_NAME = "radiance";

        using ConstructionInfo = TextureOrConstReferenceData<Vector3>;

        // Only use uv surface for now
        using Surface = UVSurface;
        // GPU Work Class will use this to specify the templated kernel
        using PrimitiveGroup        = PGroup;
        using GPUType               = GPULight;
        using HitData               = typename PGroup::HitData;
        using PrimitiveData         = typename PGroup::PrimitiveData;
        using SF                    = SurfaceFunc<Surface, HitData, PrimitiveData>;
        static constexpr SF         SurfF = SGen();

    private:
    protected:
        const CudaGPU&                  gpu;
        const PrimitiveGroup&           pg;

        DeviceMemory                    gpuLightMemory;
        const GPULight*                 dGPULights;

        // Actual Texture Allocations
        Tex2DMap                            dTextureMemory;
        std::vector<uint32_t>               textureIdList;
        uint32_t                            texDataCount;
        uint32_t                            constDataCount;
        // Texture References (or Constant Reference)
        const ConstantRef<2, Vector3>*      dConstantRadiance;
        const TextureRef<2, Vector3>*       dTextureRadiance;
        const TextureRefI<2, Vector3>**     dRadiances;
        // Temp CPU Allocations
        std::vector<uint16_t>               hMediumIds;
        std::vector<TransformId>            hTransformIds;
        std::vector<HitKey>                 hWorkKeys;
        std::vector<ConstructionInfo>       hRadianceConstructionInfo;

        GPUEndpointList				        gpuEndpointList;
        GPULightList				            gpuLightList;
        uint32_t                            lightCount;

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
        void                            SetLightLists();
        TracerError                     ConstructTextureReferences();

    public:
        // Constructors & Destructor
                                        CPULightGroupP(const GPUPrimitiveGroupI& pg,
                                                       const CudaGPU&);
                                        ~CPULightGroupP() = default;


        // Fully Implemented Interface
        const GPUEndpointList&          GPUEndpoints() const override;
        const GPULightList&             GPULights() const override;
        uint32_t                        EndpointCount() const override;
        const CudaGPU&                  GPU() const override;

        const GPULight*                 GPULightsDerived() const;
        const PrimitiveGroup&           PrimGroup() const;
};

__device__
inline GPULightP::GPULightP(const TextureRefI<2, Vector3f>& gRadiance,
                            // Base Class Related
                            uint16_t mediumIndex, HitKey hk,
                            const GPUTransformI& gTransform)
    : GPULightI(mediumIndex, hk, gTransform)
    , gRadianceRef(gRadiance)
{}

__device__
inline Vector3f GPULightP::Emit(const Vector3& wo,
                                const Vector3& pos,
                                //
                                const UVSurface& surface) const
{
    return gRadianceRef(surface.uv);
}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline CPULightGroupP<GPULight, PGroup, SGen>::CPULightGroupP(const GPUPrimitiveGroupI& pg,
                                                              const CudaGPU& gpu)
    : gpu(gpu)
    , pg(static_cast<const PGroup&>(pg))
    , lightCount(0)
    , texDataCount(0)
    , constDataCount(0)
    , dGPULights(nullptr)
    , dConstantRadiance(nullptr)
    , dTextureRadiance(nullptr)
    , dRadiances(nullptr)
{}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
SceneError CPULightGroupP<GPULight, PGroup, SGen>::InitializeCommon(const EndpointGroupDataList& lightNodes,
                                                                    const TextureNodeMap& textures,
                                                                    const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                                    const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                                    uint32_t batchId, double time,
                                                                    const std::string& scenePath)
{
    using namespace TextureFunctions;
    SceneError err = SceneError::OK;

    lightCount = static_cast<uint32_t>(lightNodes.size());
    hMediumIds.reserve(lightCount);
    hTransformIds.reserve(lightCount);
    hWorkKeys.reserve(lightCount);
    hRadianceConstructionInfo.reserve(lightCount);
    textureIdList.reserve(lightCount);

    texDataCount = 0;
    constDataCount = 0;

    uint32_t innerIndex = 0;
    for(const auto& node : lightNodes)
    {
        // Convert Ids to inner index
        uint16_t mediumIndex = static_cast<uint16_t>(mediumIdIndexPairs.at(node.mediumId));
        uint32_t transformIndex = transformIdIndexPairs.at(node.transformId);

        hMediumIds.push_back(mediumIndex);
        hTransformIds.push_back(transformIndex);
        hWorkKeys.push_back(HitKey::CombinedKey(batchId, innerIndex));

        // Load Textured Nodes
        TexturedDataNode<Vector3> radianceNode = node.node->CommonTexturedDataVector3(RADIANCE_NAME);

        ConstructionInfo constructionInfo;
        constructionInfo.isConstData = !radianceNode.isTexture;
        if(radianceNode.isTexture)
        {
            const TextureI<2>* tex;
            if((err = AllocateTexture(tex,
                                      dTextureMemory,
                                      radianceNode.texNode,
                                      textures,
                                      EdgeResolveType::WRAP,
                                      InterpolationType::LINEAR,
                                      true, true,
                                      gpu, scenePath)) != SceneError::OK)
                return err;
            constructionInfo.tex = static_cast<cudaTextureObject_t>(*tex);
            texDataCount++;
        }
        else
        {
            constructionInfo.data = radianceNode.data;
            constructionInfo.tex = 0;
            constDataCount++;
        }

        uint32_t texId = radianceNode.isTexture
                            ? radianceNode.texNode.texId
                            : std::numeric_limits<uint32_t>::max();
        textureIdList.push_back(texId);
        hRadianceConstructionInfo.push_back(constructionInfo);

        innerIndex++;
        if(innerIndex >= (1 << HitKey::IdBits))
            return SceneError::TOO_MANY_MATERIAL_IN_GROUP;
    }
    // Allocate data for texture references etc...
    DeviceMemory::AllocateMultiData(std::tie(dGPULights, dConstantRadiance,
                                             dTextureRadiance, dRadiances),
                                    gpuLightMemory,
                                    {lightCount, constDataCount,
                                     texDataCount, lightCount});

    return SceneError::OK;
}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
TracerError CPULightGroupP<GPULight, PGroup, SGen>::ConstructTextureReferences()
{
    const size_t counterCount = 2;
    DeviceMemory tempMemory;

    uint32_t* dCounters;
    ConstructionInfo* dRadianceConstructionInfo;
    DeviceMemory::AllocateMultiData(std::tie(dRadianceConstructionInfo, dCounters),
                                    tempMemory, {lightCount, counterCount});

    // Copy to GPU
    CUDA_CHECK(cudaMemset(dCounters, 0x00, sizeof(uint32_t) * counterCount));
    CUDA_CHECK(cudaMemcpy(dRadianceConstructionInfo,
                          hRadianceConstructionInfo.data(),
                          sizeof(ConstructionInfo) * lightCount,
                          cudaMemcpyHostToDevice));

    gpu.GridStrideKC_X(0, (cudaStream_t)0, lightCount,
                       GenerateEitherTexOrConstantReference<2, Vector3>,
                       const_cast<TextureRefI<2, Vector3f>**>(dRadiances),
                       const_cast<ConstantRef<2, Vector3>*>(dConstantRadiance),
                       const_cast<TextureRef<2, Vector3>*>(dTextureRadiance),
                       //
                       dCounters[0],
                       dCounters[1],
                       //
                       dRadianceConstructionInfo,
                       lightCount);

    return TracerError::OK;
}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
void CPULightGroupP<GPULight, PGroup, SGen>::SetLightLists()
{
    // Generate transform list
    for(uint32_t i = 0; i < lightCount; i++)
    {
        const auto* ptr = static_cast<const GPULightI*>(dGPULights + i);
        gpuLightList.push_back(ptr);
        gpuEndpointList.push_back(ptr);
    }
}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline const GPUEndpointList& CPULightGroupP<GPULight, PGroup, SGen>::GPUEndpoints() const
{
    return gpuEndpointList;
}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline const GPULightList& CPULightGroupP<GPULight, PGroup, SGen>::GPULights() const
{
    return gpuLightList;
}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
const GPULight* CPULightGroupP<GPULight, PGroup, SGen>::GPULightsDerived() const
{
    return dGPULights;
}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline uint32_t CPULightGroupP<GPULight, PGroup, SGen>::EndpointCount() const
{
    return lightCount;
}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline size_t CPULightGroupP<GPULight, PGroup, SGen>::UsedGPUMemory() const
{
    return gpuLightMemory.Size();
}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline size_t CPULightGroupP<GPULight, PGroup, SGen>::UsedCPUMemory() const
{
    size_t totalSize = (hMediumIds.size() * sizeof(uint16_t) +
                        hTransformIds.size() * sizeof(TransformId) +
                        hWorkKeys.size() * sizeof(HitKey));

    return totalSize;
}
template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline const CudaGPU& CPULightGroupP<GPULight, PGroup, SGen>::GPU() const
{
    return gpu;
}

template<class GPULight, class PGroup,
         SurfaceFuncGenerator<UVSurface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline const PGroup& CPULightGroupP<GPULight, PGroup, SGen>::PrimGroup() const
{
    return pg;
}
