#pragma once

#include "MetalGlassMaterialsKC.cuh"

#include "GPUMaterialP.cuh"
#include "TypeTraits.h"
#include "DeviceMemory.h"
#include "Texture.cuh"

// Metal Glass Materials
// Basic understandable naming is specifically chosen here
// for newcomers to understand what these materials correspond to
//
// For other rendering systems, "metal" is "conductor" and "glass" is "dielectric"
//
// These materials either perfect, or rough depending on the alpha parameter.
// Currently only isotropic roughness is supported. (A common single alpha value)
class MetalMat final
    : public GPUMaterialGroup<MetalMatData, UVSurface,
                              MetalMatFuncs>
{
    public:
        static const char*      TypeName() { return "MTSPlastic"; }

        using ConstantAlbedoRef = ConstantRef<2, Vector3>;
        using Texture2DRef      = TextureRef<2, Vector3>;
        using Texture2DRefI     = TextureRefI<2, Vector3>;
        using Tex2DMap          = std::map<uint32_t, std::unique_ptr<TextureI<2>>>;

        struct ConstructionInfo
        {
            bool                isConstantAlbedo = true;
            bool                hasNormalMap = false;
            Vector3f            constantAlbedo = Zero3;
            cudaTextureObject_t albedoTexture = 0;
            cudaTextureObject_t normalTexture = 0;
        };

    private:
        DeviceMemory                memory;
        // Actual Allocation of Textures
        Tex2DMap                    dTextureMemory;
        // Device Allocations of Texture References
        const ConstantAlbedoRef*    dConstDiffAlbedo;
        const Texture2DRef*         dTextureDiffAlbedoRef;
        // Aligned pointers for material access from kernel
        const Texture2DRefI**       dDiffAlbedo;
        const Texture2DRefI**       dNormal;

        // CPU Construction Info
        std::vector<ConstructionInfo> matConstructionInfo;
        std::vector<Vector2ui>        matTextureIds;

    protected:
    public:
        // Constructors & Destructor
                                MTSPlasticMat(const CudaGPU& gpu)
                                    : GPUMaterialGroup<MTSPlasticMatData, UVSurface,
                                                       MTSPlasticMatFuncs>(gpu) {}
                                ~MTSPlasticMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes,
                                           double time,
                                           const std::string& scenePath) override;
        TracerError             ConstructTextureReferences() override;

        // Material Queries
        size_t                  UsedGPUMemory() const override;
        size_t                  UsedCPUMemory() const override;
        size_t                  UsedGPUMemory(uint32_t materialId) const override;
        size_t                  UsedCPUMemory(uint32_t materialId) const override;

        uint8_t                 SampleStrategyCount() const override { return 1; };
        // No Texture
        uint8_t                 UsedTextureCount() const override;
        std::vector<uint32_t>   UsedTextureIds() const override;
};

static_assert(IsMaterialGroupClass<MTSPlasticMat>::value,
              "MTSPlasticMat is not a GPU Material Group Class.");