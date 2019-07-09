#pragma once

#include "RayLib/CosineDistribution.h"

#include "TracerLib/GPUMaterialP.cuh"

#include "SurfaceStructs.h"
#include "MaterialDataStructs.h"
#include "GIMaterialsKC.cuh"

class GIAlbedoMat final
    : public GPUMaterialGroup<TracerBasic,
                              ConstantAlbedoMatData,
                              BasicSurface,
                              GIAlbedoMatShade>
{
    public:
        static constexpr const char*    TypeName() { return "GIAlbedo"; }

    private:
        DeviceMemory            memory;

    protected:
    public:
                                GIAlbedoMat(int gpuId);
                                ~GIAlbedoMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override {return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes, double time) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time) override;

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
                                       GIAlbedoMat,
                                       GPUPrimitiveTriangle,
                                       BasicSurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
                                       GIAlbedoMat,
                                       GPUPrimitiveSphere,
                                       BasicSurfaceFromSphr>;

using GIAlbedoSphrBatch = GPUMaterialBatch<TracerBasic,
                                           GIAlbedoMat,
                                           GPUPrimitiveSphere,
                                           BasicSurfaceFromSphr>;

using GIAlbedoTriBatch = GPUMaterialBatch<TracerBasic,
                                          GIAlbedoMat,
                                          GPUPrimitiveTriangle,
                                          BasicSurfaceFromTri>;