#pragma once

#include "SampleMaterialsKC.cuh"

#include "TracerLib/MetaMaterialFunctions.cuh"
#include "TracerLib/GPUSurface.h"
#include "TracerLib/GPUMaterialP.cuh"
#include "TracerLib/TypeTraits.h"
#include "TracerLib/DeviceMemory.h"

// Light Material that constantly emits all directions
class EmissiveMat final 
    : public GPUMaterialGroup<EmissiveMatData, EmptySurface,
                              SampleEmpty<EmissiveMatData, EmptySurface>, 
                              EvaluateEmpty<EmissiveMatData, EmptySurface>,
                              EmitConstant,
                              IsEmissiveTrue<EmissiveMatData>,
                              AcquireUVEmpty<EmissiveMatData, EmptySurface>>
{
    public:
        static const char*              TypeName() { return "Emissive"; }

    private:
        DeviceMemory                    memory;

    protected:
    public:
        // Constructors & Destructor
                                EmissiveMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~EmissiveMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, 
                                                std::map<uint32_t, uint32_t> mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return memory.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsLightGroup() const override { return false; }
        bool                    IsEmissiveGroup() const override { return true; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask             CachedTextures() const { return 0; }
};

// Constant Lambert Material
class LambertMat final 
    : public GPUMaterialGroup<AlbedoMatData, BasicSurface,
                              LambertSample, LambertEvaluate,
                              EmitEmpty<AlbedoMatData, BasicSurface>,
                              IsEmissiveFalse<AlbedoMatData>,
                              AcquireUVEmpty<AlbedoMatData, BasicSurface>>
{
    public:
        static const char*              TypeName() { return "Lambert"; }

    private:
        DeviceMemory                    memory;

    protected:
    public:
        // Constructors & Destructor
                                LambertMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~LambertMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, 
                                                std::map<uint32_t, uint32_t> mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return memory.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsLightGroup() const override { return false; }
        bool                    IsEmissiveGroup() const override { return false; }

        uint8_t                 SampleStrategyCount() const { return 1; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask             CachedTextures() const { return 0; }
};

// Delta distribution reflect material
class ReflectMat final 
    : public GPUMaterialGroup<ReflectMatData, BasicSurface,
                              ReflectSample, ReflectEvaluate,
                              EmitEmpty<ReflectMatData, BasicSurface>,
                              IsEmissiveFalse<ReflectMatData>,
                              AcquireUVEmpty<ReflectMatData, BasicSurface>>
{
    public:
        static const char*              TypeName() { return "Reflect"; }

    private:
        DeviceMemory                    memory;

    protected:
    public:
        // Constructors & Destructor
                                ReflectMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~ReflectMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, 
                                                std::map<uint32_t, uint32_t> mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return memory.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

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

// Delta distribution refract material
class RefractMat final 
    : public GPUMaterialGroup<RefractMatData, BasicSurface,
                              RefractSample, RefractEvaluate,
                              EmitEmpty<RefractMatData, BasicSurface>,
                              IsEmissiveFalse<RefractMatData>,
                              AcquireUVEmpty<RefractMatData, BasicSurface>>
{
    public:
        static const char*              TypeName() { return "Refract"; }

    private:
        DeviceMemory                    memory;

    protected:
    public:
        // Constructors & Destructor
                                RefractMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~RefractMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, 
                                                std::map<uint32_t, uint32_t> mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return memory.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsLightGroup() const override { return false; }
        bool                    IsEmissiveGroup() const override { return false; }
        bool                    IsSpecularGroup() const override { return true; }

        // Post initialization
        void                    AttachGlobalMediumArray(const GPUMediumI* const*) override;

        uint8_t                 SampleStrategyCount() const { return 1; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask             CachedTextures() const { return 0; }
};

static_assert(IsTracerClass<EmissiveMat>::value,
              "EmissiveMat is not a Tracer Class.");
static_assert(IsTracerClass<LambertMat>::value,
              "LambertMat is not a Tracer Class.");
static_assert(IsTracerClass<ReflectMat>::value,
              "ReflectMat is not a Tracer Class.");
static_assert(IsTracerClass<RefractMat>::value,
              "RefractMat is not a Tracer Class.");