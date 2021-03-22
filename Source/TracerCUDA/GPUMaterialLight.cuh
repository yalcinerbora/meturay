#pragma once

#include "GPUMaterialLightKC.cuh"
#include "GPUMaterialP.cuh"
#include "GPUSurface.h"
#include "MetaMaterialFunctions.cuh"
#include "Texture.cuh"
#include "GPUDistribution.h"
#include "TypeTraits.h"

class LightMatConstant final 
    : public GPULightMaterialGroup<LightMatData, EmptySurface,
                                   SampleEmpty<LightMatData, EmptySurface>, 
                                   EvaluateEmpty<LightMatData, EmptySurface>,
                                   EmitLight,
                                   IsEmissiveTrue<LightMatData>>
{
    public:
        static const char*              TypeName() { return "LightConstant"; }

    private:
        DeviceMemory                    memory;
        std::vector<Distribution1D>     luminanceDistributions;

    public:
        // Constructors & Destructor
                                LightMatConstant(const CudaGPU& gpu) 
                                    : GPULightMaterialGroup<LightMatData, EmptySurface,
                                                            SampleEmpty<LightMatData, EmptySurface>, 
                                                            EvaluateEmpty<LightMatData, EmptySurface>,
                                                            EmitLight,
                                                            IsEmissiveTrue<LightMatData>>(gpu) {}
                                ~LightMatConstant() = default;

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
        size_t                  UsedCPUMemory() const override { return sizeof(LightMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                        IsLightGroup() const override { return true; }
        bool                        IsEmissiveGroup() const override { return true; }
        bool                        IsCameraGroup() const override { return false; }
        const GPUDistribution2D&    LuminanceDistribution(uint32_t materialId) const override;

        uint8_t                     SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                     UsedTextureCount() const { return 0; }
        std::vector<uint32_t>       UsedTextureIds() const { return std::vector<uint32_t>(); }
};

class LightMatTextured final 
    : public GPULightMaterialGroup<LightMatTexData, UVSurface,
                                   SampleEmpty<LightMatTexData, UVSurface>,
                                   EvaluateEmpty<LightMatTexData, UVSurface>,
                                   EmitLightTex,
                                   IsEmissiveTrue<LightMatTexData>>
{
    public:
        static const char*      TypeName() { return "LightTextured"; }

    private:
        DeviceMemory                        memory;
        std::vector<Texture2D<Vector4>>     textureList;
        std::vector<Distribution2D>         luminanceDistributions;

    public:
        // Constructors & Destructor
                                LightMatTextured(const CudaGPU& gpu) 
                                    : GPULightMaterialGroup<LightMatTexData, UVSurface,
                                                            SampleEmpty<LightMatTexData, UVSurface>,
                                                            EvaluateEmpty<LightMatTexData, UVSurface>,
                                                            EmitLightTex,
                                                            IsEmissiveTrue<LightMatTexData>>(gpu) {}
                                ~LightMatTextured() = default;

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
        size_t                  UsedCPUMemory() const override { return sizeof(LightMatTexData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                        IsLightGroup() const override { return true; }
        bool                        IsEmissiveGroup() const override { return true; }
        bool                        IsCameraGroup() const override { return false; }
        const GPUDistribution2D&    LuminanceDistribution(uint32_t materialId) const override;

        uint8_t                     SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                     UsedTextureCount() const { return 0; }
        std::vector<uint32_t>       UsedTextureIds() const { return std::vector<uint32_t>(); }
};

class LightMatCube final 
    : public GPULightMaterialGroup<LightMatCubeData, EmptySurface,
                                   SampleEmpty<LightMatCubeData, EmptySurface>,
                                   EvaluateEmpty<LightMatCubeData, EmptySurface>,
                                   EmitLightCube,
                                   IsEmissiveTrue<LightMatCubeData>>
{
    public:
        static const char*      TypeName() { return "LightCube"; }

    private:
        DeviceMemory                        memory;
        std::vector<TextureCube<Vector4>>   textureList;
        std::vector<Distribution2D>         luminanceDistributions;

    public:
        // Constructors & Destructor
                                LightMatCube(const CudaGPU& gpu) 
                                    : GPULightMaterialGroup<LightMatCubeData, EmptySurface,
                                                            SampleEmpty<LightMatCubeData, EmptySurface>,
                                                            EvaluateEmpty<LightMatCubeData, EmptySurface>,
                                                            EmitLightCube,
                                                            IsEmissiveTrue<LightMatCubeData>>(gpu) {}
                                ~LightMatCube() = default;

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
        size_t                  UsedCPUMemory() const override { return sizeof(LightMatTexData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                        IsLightGroup() const override { return true; }
        bool                        IsEmissiveGroup() const override { return true; }
        bool                        IsCameraGroup() const override { return false; }
        const GPUDistribution2D&    LuminanceDistribution(uint32_t materialId) const override;

        uint8_t                     SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                     UsedTextureCount() const { return 0; }
        std::vector<uint32_t>       UsedTextureIds() const { return std::vector<uint32_t>(); }
};

inline size_t LightMatTextured::UsedGPUMemory() const
{ 
    size_t totalSize = memory.Size();
    for(const auto& t : textureList)
        totalSize += t.Size();
    return totalSize;
}

inline size_t LightMatCube::UsedGPUMemory() const
{
    size_t totalSize = memory.Size();
    for(const auto& t : textureList)
        totalSize += t.Size();
    return totalSize;
}

static_assert(IsTracerClass<LightMatConstant>::value,
              "LightMatConstant is not a Tracer Class.");
static_assert(IsTracerClass<LightMatTextured>::value,
              "LightMatTextured is not a Tracer Class.");
static_assert(IsTracerClass<LightMatCube>::value,
              "LightMatCube is not a Tracer Class.");