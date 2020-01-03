#pragma once

#include "TracerLib/GPUMaterialP.cuh"

#include "SurfaceStructs.h"
#include "MaterialDataStructs.h"
#include "GIMaterialsKC.cuh"

#include "TracerLib/EmptyEventEstimator.h"
#include "TracerLib/BasicEventEstimator.h"

class BasicPathTraceMat final
    : public GPUMaterialGroup<TracerBasic,
                              BasicEventEstimator,
                              ConstantAlbedoMatData,
                              BasicSurface,
                              BasicPathTraceShade>
{
    MATERIAL_TYPE_NAME("BasicPathTrace", TracerBasic, BasicEventEstimator)

    private:
        DeviceMemory                    memory;
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
                                BasicPathTraceMat(const CudaGPU&,
                                                  const GPUEventEstimatorI&);
                                ~BasicPathTraceMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override {return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
                                                const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries
        int                     InnerId(uint32_t materialId) const override;
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; };

        size_t                  UsedGPUMemory() const override { return memory.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(ConstantAlbedoMatData); }

        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        uint8_t                 OutRayCount() const override { return 1; }
};

class LightBoundaryMat final
    : public GPUMaterialGroup<TracerBasic,
                              EmptyEventEstimator,
                              ConstantIrradianceMatData,
                              EmptySurface,
                              LightBoundaryShade>
{
    MATERIAL_TYPE_NAME("LightBoundary", TracerBasic, EmptyEventEstimator)

    private:
        DeviceMemory                    memory;
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
                                LightBoundaryMat(const CudaGPU&,
                                                 const GPUEventEstimatorI&);
                                ~LightBoundaryMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override {return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time,
                                                const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;

        // Material Queries
        int                     InnerId(uint32_t materialId) const override;
        bool                    HasCachedTextures(uint32_t materialId) const override { return false; };

        size_t                  UsedGPUMemory() const override { return memory.Size(); }
        size_t                  UsedCPUMemory() const override { return sizeof(ConstantAlbedoMatData); }

        size_t                  UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        uint8_t                 OutRayCount() const override { return 0; }
};

static_assert(IsTracerClass<BasicPathTraceMat>::value,
              "BasicPathTraceMat is not a Tracer Class.");
static_assert(IsTracerClass<LightBoundaryMat>::value,
              "LightBoundaryMat is not a Tracer Class.");

// Mat Batch Extern
extern template class GPUMaterialBatch<TracerBasic,
                                       BasicEventEstimator,
                                       BasicPathTraceMat,
                                       GPUPrimitiveTriangle,
                                       BasicSurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
                                       BasicEventEstimator,
                                       BasicPathTraceMat,
                                       GPUPrimitiveSphere,
                                       BasicSurfaceFromSphr>;

extern template class GPUMaterialBatch<TracerBasic,
                                       EmptyEventEstimator,
                                       LightBoundaryMat,
                                       GPUPrimitiveEmpty,
                                       EmptySurfaceFromEmpty>;

extern template class GPUMaterialBatch<TracerBasic,
                                       EmptyEventEstimator,
                                       LightBoundaryMat,
                                       GPUPrimitiveTriangle,
                                       EmptySurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
                                       EmptyEventEstimator,
                                       LightBoundaryMat,
                                       GPUPrimitiveSphere,
                                       EmptySurfaceFromSphr>;

using BasicPTSphereBatch = GPUMaterialBatch<TracerBasic,
                                            BasicEventEstimator,
                                            BasicPathTraceMat,
                                            GPUPrimitiveSphere,
                                            BasicSurfaceFromSphr>;

using BasicPTTriangleBatch = GPUMaterialBatch<TracerBasic,
                                              BasicEventEstimator,
                                              BasicPathTraceMat,
                                              GPUPrimitiveTriangle,
                                              BasicSurfaceFromTri>;

using LightBoundaryBatch = GPUMaterialBatch<TracerBasic,
                                            EmptyEventEstimator,
                                            LightBoundaryMat,
                                            GPUPrimitiveEmpty,
                                            EmptySurfaceFromEmpty>;

using LightBoundaryTriBatch = GPUMaterialBatch<TracerBasic,
                                               EmptyEventEstimator,
                                               LightBoundaryMat,
                                               GPUPrimitiveTriangle,
                                               EmptySurfaceFromTri>;

using LightBoundarySphrBatch = GPUMaterialBatch<TracerBasic,
                                                EmptyEventEstimator,
                                                LightBoundaryMat,
                                                GPUPrimitiveSphere,
                                                EmptySurfaceFromSphr>;

static_assert(IsTracerClass<BasicPTSphereBatch>::value,
              "BasicPTSphereBatch is not a Tracer Class.");
static_assert(IsTracerClass<BasicPTTriangleBatch>::value,
              "BasicPTTriangleBatch is not a Tracer Class.");
static_assert(IsTracerClass<LightBoundaryBatch>::value,
              "LightBoundaryBatch is not a Tracer Class.");
static_assert(IsTracerClass<LightBoundaryTriBatch>::value,
              "LightBoundaryTriBatch is not a Tracer Class.");
static_assert(IsTracerClass<LightBoundarySphrBatch>::value,
              "LightBoundarySphrBatch is not a Tracer Class.");