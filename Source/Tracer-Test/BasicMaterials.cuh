#pragma once

#include "TracerLib/GPUMaterialP.cuh"

#include "BasicTracer.cuh"
#include "SurfaceStructs.h"
#include "BasicMaterialsKC.cuh"

#include "TracerLib/TypeTraits.h"

// Unrealistic mat that directly returns an albedo regardless of wi.
// also generates invalid ray
class ConstantMat final 
    : public GPUMaterialGroupP<AlbedoMatData, EmptySurface,
                               ConstantSample, ConstantEvaluate,
                               AcquireUVEmpty<AlbedoMatData, EmptySurface>>
{
    public:
        static const char*              TypeName() { return "Constant"; }

    private:
        DeviceMemory                    memory;
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
        // Constructors & Destructor
                                ConstantMat(const CudaGPU& gpu) : GPUMaterialGroupP(gpu) {}
                                ~ConstantMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
                                                const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries
        int                     InnerId(uint32_t materialId) const override;
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return memory.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }

        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask             CachedTextures() const { return 0; }
};

class BarycentricMat final
    : public GPUMaterialGroupP<NullData, BarySurface,
                              BarycentricSample,
                              BarycentricEvaluate,
                              AcquireUVEmpty<NullData, BarySurface>
    >
{
   public:
        static const char*      TypeName() { return "Barycentric"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                BarycentricMat(const CudaGPU& gpu) : GPUMaterialGroupP(gpu) {}
                                ~BarycentricMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
                                                const std::string& scenePath) override {return SceneError::OK;}
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override {return SceneError::OK;}

        // Material Queries
        int                     InnerId(uint32_t materialId) const override { return  0; }
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }

        size_t                  UsedGPUMemory(uint32_t materialId) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask             CachedTextures() const { return 0; }
};

class SphericalMat final
    : public GPUMaterialGroupP<NullData, SphrSurface,
                               SphericalSample,
                               SphericalEvaluate,
                               AcquireUVEmpty<NullData, SphrSurface>>
{
    public:
        static const char*      TypeName() { return "Spherical"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                SphericalMat(const CudaGPU& gpu) : GPUMaterialGroupP(gpu) {}
                                ~SphericalMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
                                                const std::string& scenePath) override {return SceneError::OK;}
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override {return SceneError::OK;}

        // Material Queries
        int                     InnerId(uint32_t materialId) const override { return  0; }
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }

        size_t                  UsedGPUMemory(uint32_t materialId) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask             CachedTextures() const { return 0; }
};

static_assert(IsTracerClass<ConstantMat>::value,
              "ConstantMat is not a Tracer Class.");
static_assert(IsTracerClass<BarycentricMat>::value,
              "BarycentricMat is not a Tracer Class.");
static_assert(IsTracerClass<SphericalMat>::value,
              "SphericalMat is not a Tracer Class.");