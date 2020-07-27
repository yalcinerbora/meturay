#pragma once

#include "GPUMaterialLightKC.cuh"
#include "GPUMaterialP.cuh"
#include "SurfaceStructs.h"
#include "MetaMaterialFunctions.cuh"
#include "Texture.cuh"
#include "GPUDistribution.h"

class LightMatConstant final 
    : public GPUMaterialGroup<LightMatData, EmptySurface,
                              SampleEmpty<LightMatData, EmptySurface>, 
                              EvaluateEmpty<LightMatData, EmptySurface>,
                              EmitLight,
                              IsEmissiveTrue<LightMatData>,
                              AcquireUVEmpty<LightMatData, EmptySurface>>
    , public LightMaterialI
{
    public:
        static const char*    TypeName() { return "LightConstant"; }

    private:
        DeviceMemory                    memory;
        std::vector<Distribution1D>     luminanceDistributions;

    public:
        // Constructors & Destructor
                                LightMatConstant(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~LightMatConstant() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
                                                const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return memory.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(LightMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                        IsLightGroup() const override { return true; }
        bool                        IsEmissiveGroup() const override { return true; }
        const GPUDistribution2D&    LuminanceDistribution(uint32_t materialId) const override;

        uint8_t                     SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                     UsedTextureCount() const { return 0; }
        std::vector<uint32_t>       UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask                 CachedTextures() const { return 0; }
};

class LightMatTextured final 
    : public GPUMaterialGroup<LightMatTexData, BasicUVSurface,
                              SampleEmpty<LightMatTexData, BasicUVSurface>,
                              EvaluateEmpty<LightMatTexData, BasicUVSurface>,
                              EmitLightTex,
                              IsEmissiveTrue<LightMatTexData>,
                              AcquireUVEmpty<LightMatTexData, BasicUVSurface>>
    , public LightMaterialI
{
    public:
        static const char*      TypeName() { return "LightTextured"; }

    private:
        DeviceMemory                    memory;
        Texture2DArray<Vector4>         textureList;
        std::vector<Distribution2D>     luminanceDistributions;

    public:
        // Constructors & Destructor
                                LightMatTextured(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~LightMatTextured() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
                                                const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return memory.Size() + textureList.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(LightMatTexData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                        IsLightGroup() const override { return true; }
        bool                        IsEmissiveGroup() const override { return true; }
        const GPUDistribution2D&    LuminanceDistribution(uint32_t materialId) const override;

        uint8_t                     SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                     UsedTextureCount() const { return 0; }
        std::vector<uint32_t>       UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask                 CachedTextures() const { return 0; }
};

class LightMatCube final 
    : public GPUMaterialGroup<LightMatCubeData, EmptySurface,
                              SampleEmpty<LightMatCubeData, EmptySurface>,
                              EvaluateEmpty<LightMatCubeData, EmptySurface>,
                              EmitLightCube,
                              IsEmissiveTrue<LightMatCubeData>,
                              AcquireUVEmpty<LightMatCubeData, EmptySurface>>
    , public LightMaterialI
{
    public:
        static const char*      TypeName() { return "LightCube"; }

    private:
        DeviceMemory                    memory;
        TextureCube<Vector4>            textureList;
        std::vector<Distribution2D>     luminanceDistributions;

    public:
        // Constructors & Destructor
                                LightMatCube(const CudaGPU& gpu) : GPUMaterialGroup(gpu) {}
                                ~LightMatCube() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
                                                const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries        
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t                  UsedGPUMemory() const override { return memory.Size() + textureList.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(LightMatTexData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                        IsLightGroup() const override { return true; }
        bool                        IsEmissiveGroup() const override { return true; }
        const GPUDistribution2D&    LuminanceDistribution(uint32_t materialId) const override;

        uint8_t                     SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                     UsedTextureCount() const { return 0; }
        std::vector<uint32_t>       UsedTextureIds() const { return std::vector<uint32_t>(); }
        TextureMask                 CachedTextures() const { return 0; }
};