#pragma once

#include "GPUMaterialP.cuh"
#include "TypeTraits.h"
#include "DeviceMemory.h"
#include "DebugMaterialsKC.cuh"

class BarycentricMat final
    : public GPUMaterialGroup<NullData, BarySurface, BaryMatFuncs>
{
   public:
        static const char*      TypeName() { return "Barycentric"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                BarycentricMat(const CudaGPU& gpu)
                                    : GPUMaterialGroup<NullData, BarySurface,
                                                       BaryMatFuncs>(gpu) {}
                                ~BarycentricMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing&, double,
                                           const std::string&) override {return SceneError::OK;}

        // Material Queries
        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }
        size_t                  UsedGPUMemory(uint32_t) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t) const override { return 0; }

        // NEE Related
        bool                    CanBeSampled() const override { return false; }

        uint8_t                 SampleStrategyCount() const override { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const override { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const override { return std::vector<uint32_t>(); }
};

class SphericalMat final
    : public GPUMaterialGroup<NullData, SphrSurface, SphericalMatFuncs>
{
    public:
        static const char*      TypeName() { return "Spherical"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                SphericalMat(const CudaGPU& gpu)
                                    : GPUMaterialGroup<NullData, SphrSurface,
                                                       SphericalMatFuncs>(gpu) {}
                                ~SphericalMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing&, double,
                                           const std::string&) override {return SceneError::OK;}

        // Material Queries
        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }
        size_t                  UsedGPUMemory(uint32_t) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t) const override { return 0; }

        // NEE Related
        bool                    CanBeSampled() const override { return false; }

        uint8_t                 SampleStrategyCount() const override { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const override { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const override { return std::vector<uint32_t>(); }
};

class NormalRenderMat final
    : public GPUMaterialGroup<NullData, BasicSurface,
                              NormalRenderMatFuncs>
{
   public:
        static const char*      TypeName() { return "NormalRender"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                NormalRenderMat(const CudaGPU& gpu)
                                    : GPUMaterialGroup<NullData, BasicSurface,
                                                       NormalRenderMatFuncs>(gpu) {}
                                ~NormalRenderMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing&, double,
                                           const std::string&) override {return SceneError::OK;}

        // Material Queries
        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }
        size_t                  UsedGPUMemory(uint32_t) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t) const override { return 0; }

        // NEE Related
        bool                    CanBeSampled() const override { return false; }

        uint8_t                 SampleStrategyCount() const override { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const override { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const override { return std::vector<uint32_t>(); }
};

static_assert(IsMaterialGroupClass<BarycentricMat>::value,
              "BarycentricMat is not a GPU Material Group Class.");
static_assert(IsMaterialGroupClass<SphericalMat>::value,
              "SphericalMat is not a GPU Material Group Class.");
static_assert(IsMaterialGroupClass<NormalRenderMat>::value,
              "NormalRenderMat is not a GPU Material Group Class.");