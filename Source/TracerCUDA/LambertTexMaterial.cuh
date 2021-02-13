#pragma once

#include "LambertTexMaterialKC.cuh"
#include "MetaMaterialFunctions.cuh"
#include "GPUMaterialP.cuh"
#include "TypeTraits.h"
#include "DeviceMemory.h"
#include "Texture.cuh"

template<int C>
using Tex2DMap = std::map<uint32_t, std::unique_ptr<TextureI<2, C>>>;

// Delta distribution refract material
class LambertTexMat final 
    : public GPUMaterialGroup<LambertTMatData, UVSurface,
                              LambertTSample, LambertTEvaluate,
                              EmitEmpty<LambertTMatData, UVSurface>,
                              IsEmissiveFalse<LambertTMatData>>
{
    public:
        static const char*      TypeName() { return "LambertT"; }

        using ConstantAlbedoRef = ConstantRef<2, Vector3>;
        using Texture2DRef      = TextureRef<2, Vector3>;
        using Texture2DRefI     = TextureRefI<2, Vector3>;

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
        Tex2DMap<4>                 dTextureMemory;
        // Device Allocations of Texture References
        const ConstantAlbedoRef*    dConstAlbedo;        
        const Texture2DRef*         dTextureAlbedoRef;
        const Texture2DRef*         dTextureNormalRef;
        // Aligned pointers for material access from kernel
        const Texture2DRefI**       dAlbedo;
        const Texture2DRefI**       dNormal;

        // CPU Construction Info
        std::vector<ConstructionInfo> matConstructionInfo;

    protected:
    public:
        // Constructors & Destructor
                                LambertTexMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~LambertTexMat() = default;

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

        // Material Queries
        size_t                  UsedGPUMemory() const override;
        size_t                  UsedCPUMemory() const override;
        size_t                  UsedGPUMemory(uint32_t materialId) const override;
        size_t                  UsedCPUMemory(uint32_t materialId) const override;

        // NEE Related
        bool                    IsLightGroup() const override { return false; }
        bool                    IsEmissiveGroup() const override { return false; }
        bool                    IsSpecularGroup() const override { return true; }

        uint8_t                 SampleStrategyCount() const { return 1; };
        // No Texture
        uint8_t                 UsedTextureCount() const;
        std::vector<uint32_t>   UsedTextureIds() const;
};

static_assert(IsMaterialGroupClass<LambertTexMat>::value,
              "LambertTexMat is not a GPU Material Group Class.");
