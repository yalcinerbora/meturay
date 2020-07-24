#pragma once

#include "TracerLib/GPUMaterialP.cuh"
#include "TracerLib/SurfaceStructs.h"
#include "TracerLib/TypeTraits.h"
#include "TracerLib/MetaMaterialFunctions.cuh"

#include "BasicMaterialsKC.cuh"


// Unrealistic mat that directly returns an albedo regardless of wi.
// also generates invalid ray when sampled
class ConstantMat final 
    : public GPUMaterialGroup<AlbedoMatData, EmptySurface,
                              ConstantSample, ConstantEvaluate,
                              EmitEmpty<AlbedoMatData, EmptySurface>,
                              IsEmissiveFalse<AlbedoMatData>,
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
                                ConstantMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
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
        uint32_t                InnerId(uint32_t materialId) const override;
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return memory.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsLightGroup() const override { return false; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask             CachedTextures() const { return 0; }
};

class BarycentricMat final
    : public GPUMaterialGroup<NullData, BarySurface,
                              BarycentricSample,
                              BarycentricEvaluate,
                              EmitEmpty<NullData, BarySurface>,
                              IsEmissiveFalse<NullData>,
                              AcquireUVEmpty<NullData, BarySurface>>
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
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
                                                const std::string& scenePath) override {return SceneError::OK;}
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override {return SceneError::OK;}

        // Material Queries
        uint32_t                InnerId(uint32_t materialId) const override { return  0; }
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsLightGroup() const override { return false; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask             CachedTextures() const { return 0; }
};

class SphericalMat final
    : public GPUMaterialGroup<NullData, SphrSurface,
                              SphericalSample,
                              SphericalEvaluate,
                              EmitEmpty<NullData, SphrSurface>,
                              IsEmissiveFalse<NullData>,
                              AcquireUVEmpty<NullData, SphrSurface>>
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
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
                                                const std::string& scenePath) override {return SceneError::OK;}
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override {return SceneError::OK;}

        // Material Queries
        uint32_t                InnerId(uint32_t materialId) const override { return  0; }
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsLightGroup() const override { return false; }

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