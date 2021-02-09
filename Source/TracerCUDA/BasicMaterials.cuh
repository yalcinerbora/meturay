#pragma once

#include "GPUMaterialP.cuh"
#include "TypeTraits.h"
#include "MetaMaterialFunctions.cuh"
#include "DeviceMemory.h"
#include "BasicMaterialsKC.cuh"


// Unrealistic mat that directly returns an albedo regardless of wi.
// also generates invalid ray when sampled
class ConstantMat final 
    : public GPUMaterialGroup<AlbedoMatData, EmptySurface,
                              ConstantSample, ConstantEvaluate,
                              EmitEmpty<AlbedoMatData, EmptySurface>,
                              IsEmissiveFalse<AlbedoMatData>>
{
    public:
        static const char*              TypeName() { return "Constant"; }

    private:
        DeviceMemory                    memory;

    protected:
    public:
        // Constructors & Destructor
                                ConstantMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~ConstantMat() = default;

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
        size_t                  UsedGPUMemory() const override { return memory.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsLightGroup() const override { return false; }
        bool                    IsEmissiveGroup() const override { return false; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
};

class BarycentricMat final
    : public GPUMaterialGroup<NullData, BarySurface,
                              BarycentricSample,
                              BarycentricEvaluate,
                              EmitEmpty<NullData, BarySurface>,
                              IsEmissiveFalse<NullData>>
{
   public:
        static const char*      TypeName() { return "Barycentric"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                BarycentricMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~BarycentricMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override {return GenerateInnerIds(materialNodes);}
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override {return SceneError::OK;}

        // Material Queries
        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsLightGroup() const override { return false; }
        bool                    IsEmissiveGroup() const override { return false; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
};

class SphericalMat final
    : public GPUMaterialGroup<NullData, SphrSurface,
                              SphericalSample,
                              SphericalEvaluate,
                              EmitEmpty<NullData, SphrSurface>,
                              IsEmissiveFalse<NullData>>
{
    public:
        static const char*      TypeName() { return "Spherical"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                SphericalMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~SphericalMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, 
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override {return GenerateInnerIds(materialNodes);}
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override {return SceneError::OK;}

        // Material Queries
        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsLightGroup() const override { return false; }
        bool                    IsEmissiveGroup() const override { return false; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
};

class NormalRenderMat final
    : public GPUMaterialGroup<NullData, BasicSurface,
                              NormalSample,
                              NormalEvaluate,
                              EmitEmpty<NullData, BasicSurface>,
                              IsEmissiveFalse<NullData>>
{
   public:
        static const char*      TypeName() { return "NormalRender"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                NormalRenderMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~NormalRenderMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override {return GenerateInnerIds(materialNodes);}
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override {return SceneError::OK;}

        // Material Queries
        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsLightGroup() const override { return false; }
        bool                    IsEmissiveGroup() const override { return false; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
};

static_assert(IsMaterialGroupClass<ConstantMat>::value,
              "ConstantMat is not a GPU Material Group Class.");
static_assert(IsMaterialGroupClass<BarycentricMat>::value,
              "BarycentricMat is not a GPU Material Group Class.");
static_assert(IsMaterialGroupClass<SphericalMat>::value,
              "SphericalMat is not a GPU Material Group Class.");
static_assert(IsMaterialGroupClass<NormalRenderMat>::value,
              "NormalRenderMat is not a GPU Material Group Class.");