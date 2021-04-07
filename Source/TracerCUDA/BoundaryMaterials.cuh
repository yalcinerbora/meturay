#pragma once

#include "BoundaryMaterialsKC.cuh"
#include "GPUMaterialP.cuh"
#include "GPUSurface.h"
#include "MetaMaterialFunctions.cuh"
#include "Texture.cuh"
//#include "GPUDistribution.h"
#include "TypeTraits.h"

class BoundaryMatConstant final
    : public GPUBoundaryMaterialGroup<LightMatData, EmptySurface,
                                      SampleEmpty<LightMatData, EmptySurface>,
                                      EvaluateEmpty<LightMatData, EmptySurface>,
                                      EmitConstant,
                                      IsEmissiveTrue<LightMatData>>
{
    public:
        static const char*              TypeName() { return "BConstant"; }

    private:
        DeviceMemory                    memory;

    public:
        // Constructors & Destructor
                                BoundaryMatConstant(const CudaGPU& gpu)
                                    : GPUBoundaryMaterialGroup<LightMatData, EmptySurface,
                                                               SampleEmpty<LightMatData, EmptySurface>,
                                                               EvaluateEmpty<LightMatData, EmptySurface>,
                                                               EmitConstant,
                                                               IsEmissiveTrue<LightMatData>>(gpu) {}
                                ~BoundaryMatConstant() = default;

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

        // Light Material Interface
        TracerError                 LuminanceData(std::vector<float>& lumData,
                                                  Vector2ui& dim,
                                                  uint32_t innerId) const override;

        uint8_t                     SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                     UsedTextureCount() const { return 0; }
        std::vector<uint32_t>       UsedTextureIds() const { return std::vector<uint32_t>(); }
};

class BoundaryMatTextured final
    : public GPUBoundaryMaterialGroup<LightMatTexData, UVSurface,
                                   SampleEmpty<LightMatTexData, UVSurface>,
                                   EvaluateEmpty<LightMatTexData, UVSurface>,
                                   EmitTextured,
                                   IsEmissiveTrue<LightMatTexData>>
{
    public:
        static const char*      TypeName() { return "BTextured"; }

    private:
        DeviceMemory                        memory;
        std::vector<Texture2D<Vector4>>     textureList;

    public:
        // Constructors & Destructor
                                BoundaryMatTextured(const CudaGPU& gpu)
                                    : GPUBoundaryMaterialGroup<LightMatTexData, UVSurface,
                                                               SampleEmpty<LightMatTexData, UVSurface>,
                                                               EvaluateEmpty<LightMatTexData, UVSurface>,
                                                               EmitTextured,
                                                               IsEmissiveTrue<LightMatTexData>>(gpu) {}
                                ~BoundaryMatTextured() = default;

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
        size_t                  UsedGPUMemory(uint32_t materialId) const override;
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }
        
        // Light Material Interface
        TracerError                 LuminanceData(std::vector<float>& lumData,
                                                  Vector2ui& dim,
                                                  uint32_t innerId) const override;

        uint8_t                     SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                     UsedTextureCount() const { return 0; }
        std::vector<uint32_t>       UsedTextureIds() const { return std::vector<uint32_t>(); }
};

class BoundaryMatSkySphere final
    : public GPUBoundaryMaterialGroup<LightMatTexData, EmptySurface,
                                   SampleEmpty<LightMatTexData, EmptySurface>,
                                   EvaluateEmpty<LightMatTexData, EmptySurface>,
                                   EmitSkySphere,
                                   IsEmissiveTrue<LightMatTexData>>
{
    public:
        static const char*      TypeName() { return "BSkySphere"; }
        // Type Convenience
        using Tex2DMap          = std::map<uint32_t, std::unique_ptr<TextureI<2, 4>>>;
        using Texture2DRef      = TextureRef<2, Vector3>;

    private:
        DeviceMemory                memory;
        // Actual Allocation of Textures
        Tex2DMap                    textureMemory;
        // Temp List of Textures which will be used on
        // texture reference construction
        std::vector<Texture2DRef>    textureList;
        std::vector<Texture2DI

        

    public:
        // Constructors & Destructor
                                BoundaryMatSkySphere(const CudaGPU& gpu)
                                    : GPUBoundaryMaterialGroup<LightMatTexData, EmptySurface,
                                                               SampleEmpty<LightMatTexData, EmptySurface>,
                                                               EvaluateEmpty<LightMatTexData, EmptySurface>,
                                                               EmitSkySphere,
                                                               IsEmissiveTrue<LightMatTexData>>(gpu) {}
                                ~BoundaryMatSkySphere() = default;

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
        TracerError             ConstructTextureReferences() override;
        // Material Queries
        size_t                  UsedGPUMemory() const override;
        size_t                  UsedCPUMemory() const override { return sizeof(LightMatTexData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override;
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // Light Material Interface
        TracerError                 LuminanceData(std::vector<float>& lumData,
                                                  Vector2ui& dim,
                                                  uint32_t innerId) const override;
        
        uint8_t                     SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                     UsedTextureCount() const { return 0; }
        std::vector<uint32_t>       UsedTextureIds() const { return std::vector<uint32_t>(); }
};

inline size_t BoundaryMatTextured::UsedGPUMemory() const
{
    //size_t totalSize = memory.Size();
    //for(const auto& t : textureList)
    //    totalSize += t.Size();
    //return totalSize;
    return 0;
}

inline size_t BoundaryMatTextured::UsedGPUMemory(uint32_t materialId) const
{
    //auto loc = innerIds.find(materialId);
    //if(loc == innerIds.cend()) 
    //    return 0;

    //uint32_t innerId = loc->second;
    //size_t totalSize = textureList[innerId].Size() + sizeof(TextureRef<2, Vector3>*);    

    //return totalSize;
    return 0;
}

inline size_t BoundaryMatSkySphere::UsedGPUMemory() const
{
    //size_t totalSize = memory.Size();
    //for(const auto& t : textureMemory)
    //    totalSize += .Size();
    //return totalSize;
    return 0;
}

inline size_t BoundaryMatSkySphere::UsedGPUMemory(uint32_t materialId) const
{
    //auto loc = innerIds.find(materialId);
    //if(loc == innerIds.cend())
    //    return 0;

    //uint32_t innerId = loc->second;
    //size_t totalSize = textureList[innerId].Size() + sizeof(TextureRef<2, Vector3>*);
    //return totalSize;
    return 0;
}

static_assert(IsTracerClass<BoundaryMatConstant>::value,
              "BoundaryMatConstant is not a Tracer Class.");
static_assert(IsTracerClass<BoundaryMatTextured>::value,
              "BoundaryMatTextured is not a Tracer Class.");
static_assert(IsTracerClass<BoundaryMatSkySphere>::value,
              "BoundaryMatSkySphere is not a Tracer Class.");