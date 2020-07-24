#pragma once

#include "SampleMaterialsKC.cuh"

#include "TracerLib/MetaMaterialFunctions.cuh"
#include "TracerLib/SurfaceStructs.h"
#include "TracerLib/GPUMaterialP.cuh"
#include "TracerLib/TypeTraits.h"

// Light Material that constantly emits all directions
class EmissiveMat final 
    : public GPUMaterialGroup<EmissiveMatData, EmptySurface,
                              SampleEmpty<EmissiveMatData, EmptySurface>, 
                              EvaluateEmpty<EmissiveMatData, EmptySurface>,
                              EmitConstant<EmptySurface>,
                              IsEmissiveTrue<EmissiveMatData>,
                              AcquireUVEmpty<EmissiveMatData, EmptySurface>>
{
    public:
        static const char*              TypeName() { return "Emissive"; }

    private:
        DeviceMemory                    memory;
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
        // Constructors & Destructor
                                EmissiveMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~EmissiveMat() = default;

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
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
        // Constructors & Destructor
                                LambertMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~LambertMat() = default;

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
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
        // Constructors & Destructor
                                ReflectMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~ReflectMat() = default;

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
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
        // Constructors & Destructor
                                RefractMat(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~RefractMat() = default;

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

        uint8_t                 SampleStrategyCount() const { return 1; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask             CachedTextures() const { return 0; }
};