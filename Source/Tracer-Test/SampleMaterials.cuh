//#pragma once
//
//#include "TracerLib/GPUMaterialP.cuh"
//
//#include "BasicTracer.cuh"
//#include "SurfaceStructs.h"
//#include "MaterialDataStructs.h"
//#include "GIMaterialsKC.cuh"

#include "TracerLib/GPUMaterialP.cuh"

#include "SurfaceStructs.h"
#include "SampleMaterialsKC.cuh"
#include "BasicMaterialsKC.cuh"

#include "TracerLib/TypeTraits.h"

template <class Surface>
Vector3 EmitConstant(// Input
                  const Vector3& wo,
                  const Vector3& pos,
                  const GPUMedium& m,
                  //
                  const Surface& surface,
                  const TexCoords* uvs,
                  // Constants
                  const EmissiveMatData& matData,
                  const HitKey::Type& matId)
{
    return matData.dAlbedo[matId];
}

// Light Material that constantly emits all directions
class EmissiveMat final 
    : public GPUMaterialGroup<EmissiveMatData, EmptySurface,
                              ConstantSample, ConstantEvaluate,
                              EmitConstant<EmptySurface>,
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

// Constant Lambert Material
class LambertMat final 
    : public GPUMaterialGroup<AlbedoMatData, EmptySurface,
                              ConstantSample, ConstantEvaluate,
                              EmitEmpty<AlbedoMatData, EmptySurface>,
                              AcquireUVEmpty<AlbedoMatData, EmptySurface>>
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

// Delta distribution reflect material
class ReflectMat final 
    : public GPUMaterialGroup<ReflectMatData, BasicSurface,
                              ReflectSample, ReflectEvaluate,
                              EmitEmpty<ReflectMatData, BasicSurface>,
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

// Delta distribution refract material
class RefractMat final 
    : public GPUMaterialGroup<RefractMatData, BasicSurface,
                              RefractSample, RefractEvaluate,
                              EmitEmpty<RefractMatData, BasicSurface>,
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


//class BasicPathTraceMat final
//    : public GPUMaterialGroup<TracerBasic,
//                              GPUEventEstimatorBasic,
//                              AlbedoMatData,
//                              BasicSurface,
//                              BasicDiffusePTShade>
//{
//    MATERIAL_TYPE_NAME("BasicPathTrace", TracerBasic, GPUEventEstimatorBasic)
//
//    private:
//        DeviceMemory                    memory;
//        std::map<uint32_t, uint32_t>    innerIds;
//
//    protected:
//    public:
//                                BasicPathTraceMat(const CudaGPU& gpu, const GPUEventEstimatorI& e)
//                                    : GPUMaterialGroup(gpu, e) {}
//                                ~BasicPathTraceMat() = default;
//
//        // Interface
//        // Type (as string) of the primitive group
//        const char*             Type() const override {return TypeName(); }
//        // Allocates and Generates Data
//        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
//                                                const std::string& scenePath) override;
//        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
//                                           const std::string& scenePath) override;
//
//        // Material Queries
//        int                     InnerId(uint32_t materialId) const override { return innerIds.at(materialId); }
//        bool                    HasCachedTextures(uint32_t materialId) const override { return false; };
//
//        size_t                  UsedGPUMemory() const override { return memory.Size(); }
//        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }
//
//        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
//        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }
//
//        uint8_t                 OutRayCount() const override { return BASICPT_MAX_OUT_RAY; }
//};
//
//class LightBoundaryMat final
//    : public GPUMaterialGroup<IrradianceMatData,
//                              EmptySurface,
//                              LightBoundaryShade>
//{
//    MATERIAL_TYPE_NAME("LightBoundary", TracerBasic, GPUEventEstimatorBasic)
//
//    private:
//        DeviceMemory                    memory;
//        std::map<uint32_t, uint32_t>    innerIds;
//
//    protected:
//    public:
//                                LightBoundaryMat(const CudaGPU& gpu, const GPUEventEstimatorI& e)
//                                    : GPUMaterialGroup(gpu, e) {}
//                                ~LightBoundaryMat() = default;
//
//        // Interface
//        // Type (as string) of the primitive group
//        const char*             Type() const override {return TypeName(); }
//        // Allocates and Generates Data
//        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
//                                                const std::string& scenePath) override;
//        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
//                                           const std::string& scenePath) override;
//
//        // Material Queries
//        int                     InnerId(uint32_t materialId) const override { return innerIds.at(materialId); }
//        bool                    HasCachedTextures(uint32_t materialId) const override { return false; };
//
//        size_t                  UsedGPUMemory() const override { return memory.Size(); }
//        size_t                  UsedCPUMemory() const override { return sizeof(IrradianceMatData); }
//
//        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
//        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }
//
//        uint8_t                 OutRayCount() const override { return 0; }
//};
//
//class BasicReflectPTMat final 
//    : public GPUMaterialGroup<TracerBasic,
//                              GPUEventEstimatorBasic,
//                              ReflectMatData,
//                              BasicSurface,
//                              BasicReflectPTShade>
//{
//   MATERIAL_TYPE_NAME("BasicReflectPT", TracerBasic, GPUEventEstimatorBasic)
//
//    private:
//        DeviceMemory                    memory;
//        std::map<uint32_t, uint32_t>    innerIds;
//
//    protected:
//    public:
//                                BasicReflectPTMat(const CudaGPU& gpu, const GPUEventEstimatorI& e) 
//                                    : GPUMaterialGroup(gpu, e) {}
//                                ~BasicReflectPTMat() = default;
//
//        // Interface
//        // Type (as string) of the primitive group
//        const char*             Type() const override {return TypeName(); }
//        // Allocates and Generates Data
//        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
//                                                const std::string& scenePath) override;
//        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
//                                           const std::string& scenePath) override;
//
//        // Material Queries
//        int                     InnerId(uint32_t materialId) const override { return innerIds.at(materialId); }
//        bool                    HasCachedTextures(uint32_t materialId) const override { return false; };
//
//        size_t                  UsedGPUMemory() const override { return memory.Size(); }
//        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }
//
//        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
//        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }
//
//        uint8_t                 OutRayCount() const override { return REFLECTPT_MAX_OUT_RAY; }
//};
//
//class BasicRefractPTMat final
//    : public GPUMaterialGroup<TracerBasic,
//                              GPUEventEstimatorBasic,
//                              RefractMatData,
//                              BasicSurface,
//                              BasicRefractPTShade>
//{
//       MATERIAL_TYPE_NAME("BasicRefractPT", TracerBasic, GPUEventEstimatorBasic)
//
//    private:
//        DeviceMemory                    memory;
//        std::map<uint32_t, uint32_t>    innerIds;
//
//    protected:
//    public:
//                                BasicRefractPTMat(const CudaGPU& gpu, const GPUEventEstimatorI& e)
//                                    : GPUMaterialGroup(gpu, e) {}
//                                ~BasicRefractPTMat() = default;
//
//        // Interface
//        // Type (as string) of the primitive group
//        const char*             Type() const override {return TypeName(); }
//        // Allocates and Generates Data
//        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
//                                                const std::string& scenePath) override;
//        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
//                                           const std::string& scenePath) override;
//
//        // Material Queries
//        int                     InnerId(uint32_t materialId) const override { return innerIds.at(materialId); }
//        bool                    HasCachedTextures(uint32_t materialId) const override { return false; };
//
//        size_t                  UsedGPUMemory() const override { return memory.Size(); }
//        size_t                  UsedCPUMemory() const override { return sizeof(AlbedoMatData); }
//
//        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
//        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }
//
//        uint8_t                 OutRayCount() const override { return REFRACTPT_MAX_OUT_RAY; }
//};
//
//static_assert(IsTracerClass<BasicPathTraceMat>::value,
//              "BasicPathTraceMat is not a Tracer Class.");
//static_assert(IsTracerClass<LightBoundaryMat>::value,
//              "LightBoundaryMat is not a Tracer Class.");
//static_assert(IsTracerClass<BasicReflectPTMat>::value,
//              "BasicReflectPTMat is not a Tracer Class.");
//static_assert(IsTracerClass<BasicRefractPTMat>::value,
//              "BasicRefractPTMat is not a Tracer Class.");
//
//#define DECLARE_MATH_BATCH(NAME, TRACER, ESTIMATOR, MATERIAL, PRIMITIVE, SURF_FUNC)\
//    extern template class GPUMaterialBatch<TRACER, ESTIMATOR, MATERIAL, PRIMITIVE, SURF_FUNC>;\
//    using NAME = GPUMaterialBatch<TRACER, ESTIMATOR, MATERIAL, PRIMITIVE, SURF_FUNC>;\
//    static_assert(IsTracerClass<NAME>::value, #NAME" is not a Tracer Class.");
//
//#define DEFINE_MATH_BATCH(NAME, TRACER, ESTIMATOR, MATERIAL, PRIMITIVE, SURF_FUNC) \
//    template class GPUMaterialBatch<TRACER, ESTIMATOR, MATERIAL, PRIMITIVE, SURF_FUNC>;
//
//
//// Diffuse
//DECLARE_MATH_BATCH(BasicPTTriangleBatch, TracerBasic, GPUEventEstimatorBasic,
//                   BasicPathTraceMat, GPUPrimitiveTriangle, BasicSurfaceFromTri);
//
//DECLARE_MATH_BATCH(BasicPTSphereBatch, TracerBasic, GPUEventEstimatorBasic,
//                   BasicPathTraceMat, GPUPrimitiveSphere, BasicSurfaceFromSphr);
//
//// Light
//DECLARE_MATH_BATCH(LightBoundaryBatch, TracerBasic, GPUEventEstimatorBasic,
//                   LightBoundaryMat, GPUPrimitiveEmpty, EmptySurfaceFromEmpty);
//
//DECLARE_MATH_BATCH(LightBoundaryTriBatch, TracerBasic, GPUEventEstimatorBasic,
//                   LightBoundaryMat, GPUPrimitiveTriangle, EmptySurfaceFromTri);
//
//DECLARE_MATH_BATCH(LightBoundarySphrBatch, TracerBasic, GPUEventEstimatorBasic,
//                   LightBoundaryMat, GPUPrimitiveSphere, EmptySurfaceFromSphr);
//
//// Reflect
//DECLARE_MATH_BATCH(ReflectPTTriangleBatch, TracerBasic, GPUEventEstimatorBasic,
//                   BasicReflectPTMat, GPUPrimitiveTriangle, BasicSurfaceFromTri);
//
//DECLARE_MATH_BATCH(ReflectPTSphereBatch, TracerBasic, GPUEventEstimatorBasic,
//                   BasicReflectPTMat, GPUPrimitiveSphere, BasicSurfaceFromSphr);
//
//// Refract
//DECLARE_MATH_BATCH(RefractPTTriangleBatch, TracerBasic, GPUEventEstimatorBasic,
//                   BasicRefractPTMat, GPUPrimitiveTriangle, BasicSurfaceFromTri);
//
//DECLARE_MATH_BATCH(RefractPTSphereBatch, TracerBasic, GPUEventEstimatorBasic,
//                   BasicRefractPTMat, GPUPrimitiveSphere, BasicSurfaceFromSphr);