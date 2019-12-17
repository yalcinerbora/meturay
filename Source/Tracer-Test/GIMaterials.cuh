#pragma once

#include "RayLib/CosineDistribution.h"

#include "TracerLib/GPUMaterialP.cuh"

#include "SurfaceStructs.h"
#include "MaterialDataStructs.h"
#include "GIMaterialsKC.cuh"

using ConstantIrradianceMatData = ConstantAlbedoMatData;

class BasicPathTraceMat final
    : public GPUMaterialGroup<TracerBasic,
                              ConstantAlbedoMatData,
                              BasicSurface,
                              BasicPathTraceShade>
{
    public:
        static constexpr const char*    TypeName() { return "BasicPathTrace"; }

    private:
        DeviceMemory                    memory;
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
                                BasicPathTraceMat(int gpuId);
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
                              ConstantIrradianceMatData,
                              EmptySurface,
                              LightBoundaryShade>
{
    public:
        static constexpr const char*    TypeName() { return "LightBoundary"; }

    private:
        DeviceMemory                    memory;
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
                                LightBoundaryMat(int gpuId);
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

        uint8_t                 OutRayCount() const override { return 1; }
};

// Mat Batch Extern
extern template class GPUMaterialBatch<TracerBasic,
                                       BasicPathTraceMat,
                                       GPUPrimitiveTriangle,
                                       BasicSurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
                                       BasicPathTraceMat,
                                       GPUPrimitiveSphere,
                                       BasicSurfaceFromSphr>;

extern template class GPUMaterialBatch<TracerBasic,
                                       LightBoundaryMat,
                                       GPUPrimitiveEmpty,
                                       EmptySurfaceFromEmpty>;

using BasicPTSphereBatch = GPUMaterialBatch<TracerBasic,
                                            BasicPathTraceMat,
                                            GPUPrimitiveSphere,
                                            BasicSurfaceFromSphr>;

using BasicPTTriangleBatch = GPUMaterialBatch<TracerBasic,
                                              BasicPathTraceMat,
                                              GPUPrimitiveTriangle,
                                              BasicSurfaceFromTri>;

using LightBoundaryBatch = GPUMaterialBatch<TracerBasic,
                                            LightBoundaryMat,
                                            GPUPrimitiveEmpty,
                                            EmptySurfaceFromEmpty>;