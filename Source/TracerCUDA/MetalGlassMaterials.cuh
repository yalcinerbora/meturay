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
        static const char*      TypeName() { return "Metal"; }

    private:
        DeviceMemory            memory;

    protected:
    public:
        // Constructors & Destructor
                                MetalMat(const CudaGPU& gpu)
                                    : GPUMaterialGroup<MetalMatData, UVSurface,
                                                       MetalMatFuncs>(gpu) {}
                                ~MetalMat() = default;

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

static_assert(IsMaterialGroupClass<MetalMat>::value,
              "MetalMat is not a GPU Material Group Class.");