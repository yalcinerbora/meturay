#pragma once
/**

Unreal Engine Physically Based Shader Implementation

https://www.semanticscholar.org/paper/Real-Shading-in-Unreal-Engine-4-by-Karis/91ee695f6a64d8508817fa3c0203d4389c462536

Implementation of Microfacet Shader definition of the Unreal Engine

*/

#include "UnrealMaterialKC.cuh"

#include "MetaMaterialFunctions.cuh"
#include "GPUSurface.h"
#include "GPUMaterialP.cuh"
#include "TypeTraits.h"
#include "DeviceMemory.h"

class UnrealMat final
    : public GPUMaterialGroup<UnrealMatData, UVSurface,
                              UnrealSample, UnrealEvaluate,
                              EmitEmpty<UnrealMatData, UVSurface>,
                              IsEmissiveFalse<UnrealMatData>>
{
    public:
        static const char*          TypeName() { return "Unreal"; }

        // Node Property Names
        static constexpr const char* ALBEDO = "albedo";
        static constexpr const char* ROUGHNESS = "roughness";
        static constexpr const char* METALLIC = "metallic";

    private:
        DeviceMemory                memory;
            protected:
    public:
        // Constructors & Destructor
                                UnrealMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~UnrealMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
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

static_assert(IsMaterialGroupClass<UnrealMat>::value,
              "UnrealMat is not a GPU Material Group Class.");