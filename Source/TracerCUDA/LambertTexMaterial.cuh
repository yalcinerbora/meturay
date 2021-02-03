#pragma once

#include "LambertTexMaterialKC.cuh"
#include "MetaMaterialFunctions.cuh"
#include "GPUMaterialP.cuh"
#include "TypeTraits.h"
#include "DeviceMemory.h"
#include "Texture.cuh"


template<class T>
using Tex2DMap = std::map<uint32_t, Texture<2, T>>;

// Delta distribution refract material
class LambertTexMat final 
    : public GPUMaterialGroup<LambertTMatData, UVSurface,
                              LambertTSample, LambertTEvaluate,
                              EmitEmpty<LambertTMatData, UVSurface>,
                              IsEmissiveFalse<LambertTMatData>>
{
    public:
        static const char*      TypeName() { return "LambertT"; }

    private:
        DeviceMemory            memory;
        Tex2DMap<Vector3>       textureMemory;

        Vector3f*               dConstAlbedo;

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
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override;
        size_t                  UsedCPUMemory() const override { return sizeof(LambertTMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override;
        size_t                  UsedCPUMemory(uint32_t materialId) const override;

        // NEE Related
        bool                    IsLightGroup() const override { return false; }
        bool                    IsEmissiveGroup() const override { return false; }
        bool                    IsSpecularGroup() const override { return true; }

        uint8_t                 SampleStrategyCount() const { return 1; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask             CachedTextures() const { return 0; }
};

static_assert(IsMaterialGroupClass<LambertTexMat>::value,
              "LambertTexMat is not a GPU Material Group Class.");
