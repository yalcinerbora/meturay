#pragma once

#include "SimpleMaterialsKC.cuh"

#include "MetaMaterialFunctions.cuh"
#include "GPUMaterialP.cuh"
#include "TypeTraits.h"
#include "DeviceMemory.h"

//// Light Material that constantly emits all directions
//class EmissiveMat final
//    : public GPUMaterialGroup<EmissiveMatData, EmptySurface,
//                              SampleEmpty<EmissiveMatData, EmptySurface>,
//                              EvaluateEmpty<EmissiveMatData, EmptySurface>,
//                              EmitConstant,
//                              IsEmissiveTrue<EmissiveMatData>>
//{
//    public:
//        static const char*              TypeName() { return "Emissive"; }
//
//    private:
//        DeviceMemory                    memory;
//
//    protected:
//    public:
//        // Constructors & Destructor
//                                EmissiveMat(const CudaGPU& gpu)
//                                    : GPUMaterialGroup<EmissiveMatData, EmptySurface,
//                                                       SampleEmpty<EmissiveMatData, EmptySurface>,
//                                                       EvaluateEmpty<EmissiveMatData, EmptySurface>,
//                                                       EmitConstant,
//                                                       IsEmissiveTrue<EmissiveMatData>>(gpu) {}
//                                ~EmissiveMat() = default;
//
//        // Interface
//        // Type (as string) of the primitive group
//        const char*             Type() const override { return TypeName(); }
//        // Allocates and Generates Data
//        SceneError              InitializeGroup(const NodeListing& materialNodes,
//                                                const TextureNodeMap& textures,
//                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
//                                                double time, const std::string& scenePath) override;
//        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
//                                           const std::string& scenePath) override;
//
//        // Material Queries
//        size_t                  UsedGPUMemory() const override { return memory.Size(); }
//        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }
//        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
//        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }
//
//        // NEE Related
//        bool                    IsLightGroup() const override { return false; }
//        bool                    IsEmissiveGroup() const override { return true; }
//        bool                    IsCameraGroup() const override { return false; }
//
//        uint8_t                 SampleStrategyCount() const { return 0; };
//        // No Texture
//        uint8_t                 UsedTextureCount() const { return 0; }
//        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
//};

// Constant Lambert Material
class LambertCMat final
    : public GPUMaterialGroup<AlbedoMatData, BasicSurface,
                              LambertCSample, LambertCEvaluate,
                              EmitEmpty<AlbedoMatData, BasicSurface>,
                              IsEmissiveFalse<AlbedoMatData>>
{
    public:
        static const char*              TypeName() { return "LambertC"; }

    private:
        DeviceMemory                    memory;

    protected:
    public:
        // Constructors & Destructor
                                LambertCMat(const CudaGPU& gpu)
                                    : GPUMaterialGroup<AlbedoMatData, BasicSurface,
                                                       LambertCSample, LambertCEvaluate,
                                                       EmitEmpty<AlbedoMatData, BasicSurface>,
                                                       IsEmissiveFalse<AlbedoMatData>>(gpu) {}
                                ~LambertCMat() = default;

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

        uint8_t                 SampleStrategyCount() const { return 1; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
};

// Delta distribution reflect material
class ReflectMat final
    : public GPUMaterialGroup<ReflectMatData, BasicSurface,
                              ReflectSample, ReflectEvaluate,
                              EmitEmpty<ReflectMatData, BasicSurface>,
                              IsEmissiveFalse<ReflectMatData>>
{
    public:
        static const char*              TypeName() { return "Reflect"; }

    private:
        DeviceMemory                    memory;

    protected:
    public:
        // Constructors & Destructor
                                ReflectMat(const CudaGPU& gpu)
                                    : GPUMaterialGroup<ReflectMatData, BasicSurface,
                                                       ReflectSample, ReflectEvaluate,
                                                       EmitEmpty<ReflectMatData, BasicSurface>,
                                                       IsEmissiveFalse<ReflectMatData>>(gpu) {}
                                ~ReflectMat() = default;

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
        size_t                  UsedCPUMemory() const override { return sizeof(ReflectMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector4); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    CanBeSampled() const { return false; }

        uint8_t                 SampleStrategyCount() const { return 1; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
};

// Delta distribution refract material
class RefractMat final
    : public GPUMaterialGroup<RefractMatData, BasicSurface,
                              RefractSample, RefractEvaluate,
                              EmitEmpty<RefractMatData, BasicSurface>,
                              IsEmissiveFalse<RefractMatData>>
{
    public:
        static const char*      TypeName() { return "Refract"; }

    private:
        DeviceMemory            memory;

    protected:
    public:
        // Constructors & Destructor
                                RefractMat(const CudaGPU& gpu)
                                    : GPUMaterialGroup<RefractMatData, BasicSurface,
                                                       RefractSample, RefractEvaluate,
                                                       EmitEmpty<RefractMatData, BasicSurface>,
                                                       IsEmissiveFalse<RefractMatData>>(gpu) {}
                                ~RefractMat() = default;

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
        size_t                  UsedCPUMemory() const override { return sizeof(RefractMatData); }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f) + sizeof(uint32_t); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    CanBeSampled() const { return false; }

        // Post initialization
        void                    AttachGlobalMediumArray(const GPUMediumI* const*,
                                                        uint32_t baseMediumIndex) override;

        uint8_t                 SampleStrategyCount() const { return 1; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
};

static_assert(IsMaterialGroupClass<LambertCMat>::value,
              "LambertCMat is not a GPU Material Group Class.");
static_assert(IsMaterialGroupClass<ReflectMat>::value,
              "ReflectMat is not a GPU Material Group Class.");
static_assert(IsMaterialGroupClass<RefractMat>::value,
              "RefractMat is not a GPU Material Group Class.");