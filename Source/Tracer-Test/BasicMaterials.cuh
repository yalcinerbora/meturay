#pragma once

#include "TracerLib/GPUMaterialP.cuh"

#include "BasicTracer.cuh"
#include "SurfaceStructs.h"
#include "BasicMaterialsKC.cuh"
#include "TracerLib/TypeTraits.h"

class BasicMat final
    : public GPUMaterialGroup<TracerBasic,
                              GPUEventEstimatorEmpty,
                              AlbedoMatData,
                              EmptySurface,
                              BasicMatShade>
{
    MATERIAL_TYPE_NAME("BasicMat", TracerBasic, GPUEventEstimatorEmpty)

    private:
        DeviceMemory                    memory;
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
        // Constructors & Destructor
                        BasicMat(const CudaGPU&,
                                 const GPUEventEstimatorI&);
                        ~BasicMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*     Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError      InitializeGroup(const NodeListing& materialNodes, double time,
                                        const std::string& scenePath) override;
        SceneError      ChangeTime(const NodeListing& materialNodes, double time,
                                   const std::string& scenePath) override;

        // Material Queries
        int            InnerId(uint32_t materialId) const override;
        bool           HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t         UsedGPUMemory() const override { return memory.Size(); }
        size_t         UsedCPUMemory() const override { return sizeof(AlbedoMatData); }

        size_t         UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t         UsedCPUMemory(uint32_t materialId) const override { return 0; }

        uint8_t        OutRayCount() const override { return 0; }
};

class BarycentricMat final
    : public GPUMaterialGroup<TracerBasic,
                              GPUEventEstimatorEmpty,
                              NullData,
                              BarySurface,
                              BaryMatShade>
{
    MATERIAL_TYPE_NAME("BarycentricMat", TracerBasic, GPUEventEstimatorEmpty)

    private:
    protected:
    public:
        // Constructors & Destructor
                        BarycentricMat(const CudaGPU&,
                                       const GPUEventEstimatorI&);
                        ~BarycentricMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*     Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError      InitializeGroup(const NodeListing& materialNodes, double time,
                                        const std::string& scenePath) override;
        SceneError      ChangeTime(const NodeListing& materialNodes, double time,
                                   const std::string& scenePath) override;

        // Material Queries
        int             InnerId(uint32_t materialId) const override;
        bool            HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t          UsedGPUMemory() const override { return 0; }
        size_t          UsedCPUMemory() const override { return 0; }

        size_t          UsedGPUMemory(uint32_t materialId) const override { return 0; }
        size_t          UsedCPUMemory(uint32_t materialId) const override { return 0; }

        uint8_t         OutRayCount() const override { return 0; }
};

class SphericalMat final
    : public GPUMaterialGroup<TracerBasic,
                              GPUEventEstimatorEmpty,
                              NullData,
                              SphrSurface,
                              SphrMatShade>
{
    MATERIAL_TYPE_NAME("SphericalMat", TracerBasic, GPUEventEstimatorEmpty)

    private:
    protected:
    public:
        // Constructors & Destructor
                        SphericalMat(const CudaGPU&,
                                     const GPUEventEstimatorI&);
                        ~SphericalMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*     Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError      InitializeGroup(const NodeListing& materialNodes, double time,
                                        const std::string& scenePath) override;
        SceneError      ChangeTime(const NodeListing& materialNodes, double time,
                                   const std::string& scenePath) override;

        // Material Queries
        int             InnerId(uint32_t materialId) const override;
        bool            HasCachedTextures(uint32_t materialId) const override { return false; }

        size_t          UsedGPUMemory() const override { return 0; }
        size_t          UsedCPUMemory() const override { return 0; }

        size_t          UsedGPUMemory(uint32_t materialId) const override { return 0; }
        size_t          UsedCPUMemory(uint32_t materialId) const override { return 0; }

        uint8_t         OutRayCount() const override { return 0; }
};

static_assert(IsTracerClass<BasicMat>::value,
              "BasicMat is not a Tracer Class.");
static_assert(IsTracerClass<BarycentricMat>::value,
              "BarycentricMat is not a Tracer Class.");
static_assert(IsTracerClass<SphericalMat>::value,
              "SphericalMat is not a Tracer Class.");

// Material Batches
extern template class GPUMaterialBatch<TracerBasic,
                                       GPUEventEstimatorEmpty,
                                       BasicMat,
                                       GPUPrimitiveEmpty,
                                       EmptySurfaceFromEmpty>;

extern template class GPUMaterialBatch<TracerBasic,
                                       GPUEventEstimatorEmpty,
                                       BasicMat,
                                       GPUPrimitiveSphere,
                                       EmptySurfaceFromSphr>;

extern template class GPUMaterialBatch<TracerBasic,
                                       GPUEventEstimatorEmpty,
                                       BasicMat,
                                       GPUPrimitiveTriangle,
                                       EmptySurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
                                       GPUEventEstimatorEmpty,
                                       BarycentricMat,
                                       GPUPrimitiveTriangle,
                                       BarySurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
                                       GPUEventEstimatorEmpty,
                                       SphericalMat,
                                       GPUPrimitiveSphere,
                                       SphrSurfaceFromSphr>;

using BasicMatBatch = GPUMaterialBatch<TracerBasic,
                                       GPUEventEstimatorEmpty,
                                       BasicMat,
                                       GPUPrimitiveEmpty,
                                       EmptySurfaceFromEmpty>;

using BasicMatTriBatch = GPUMaterialBatch<TracerBasic,
                                          GPUEventEstimatorEmpty,
                                          BasicMat,
                                          GPUPrimitiveTriangle,
                                          EmptySurfaceFromTri>;

using BasicMatSphrBatch = GPUMaterialBatch<TracerBasic,
                                           GPUEventEstimatorEmpty,
                                           BasicMat,
                                           GPUPrimitiveSphere,
                                           EmptySurfaceFromSphr>;

using BarycentricMatTriBatch = GPUMaterialBatch<TracerBasic,
                                                GPUEventEstimatorEmpty,
                                                BarycentricMat,
                                                GPUPrimitiveTriangle,
                                                BarySurfaceFromTri>;

using SphericalMatSphrBatch = GPUMaterialBatch<TracerBasic,
                                               GPUEventEstimatorEmpty,
                                               SphericalMat,
                                               GPUPrimitiveSphere,
                                               SphrSurfaceFromSphr>;

static_assert(IsTracerClass<BasicMatBatch>::value,
              "BasicMatEmptyBatch is not a Tracer Class.");
static_assert(IsTracerClass<BasicMatTriBatch>::value,
              "BasicMatTriBatch is not a Tracer Class.");
static_assert(IsTracerClass<BasicMatSphrBatch>::value,
              "BasicMatSphrBatch is not a Tracer Class.");
static_assert(IsTracerClass<BarycentricMatTriBatch>::value,
              "BarycentricMatTriBatch is not a Tracer Class.");
static_assert(IsTracerClass<SphericalMatSphrBatch>::value,
              "SphericalMatSphrBatch is not a Tracer Class.");