#pragma once

#include "TracerLib/GPUMaterialP.cuh"

#include "SurfaceStructs.h"
#include "BasicMaterialsKC.cuh"
#include "TracerLib/TypeTraits.h"

class BasicMat final
    : public GPUMaterialGroup<TracerBasic,
                              ConstantAlbedoMatData,
                              EmptySurface,
                              BasicMatShade>
{
    public:
        static constexpr const char*    TypeName() { return "BasicMat"; }

    private:
        DeviceMemory                    memory;
        std::map<uint32_t, uint32_t>    innerIds;

    protected:
    public:
        // Constructors & Destructor
                        BasicMat(int gpuId);
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
        size_t         UsedCPUMemory() const override { return sizeof(ConstantAlbedoMatData); }

        size_t         UsedGPUMemory(uint32_t materialId) const override { return sizeof(Vector3f); }
        size_t         UsedCPUMemory(uint32_t materialId) const override { return 0; }

        uint8_t        OutRayCount() const override { return 0; }
};

class BarycentricMat final
    : public GPUMaterialGroup<TracerBasic,
                              NullData,
                              BarySurface,
                              BaryMatShade>
{
    public:
        static constexpr const char*    TypeName() { return "BarycentricMat"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                        BarycentricMat(int gpuId);
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
                              NullData,
                              SphrSurface,
                              SphrMatShade>
{
    public:
        static constexpr const char*    TypeName() { return "SphericalMat"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                        SphericalMat(int gpuId);
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
                                       BasicMat,
                                       GPUPrimitiveEmpty,
                                       EmptySurfaceFromEmpty>;

extern template class GPUMaterialBatch<TracerBasic,
                                       BasicMat,
                                       GPUPrimitiveSphere,
                                       EmptySurfaceFromSphr>;

extern template class GPUMaterialBatch<TracerBasic,
                                       BasicMat,
                                       GPUPrimitiveTriangle,
                                       EmptySurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
                                       BarycentricMat,
                                       GPUPrimitiveTriangle,
                                       BarySurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
                                       SphericalMat,
                                       GPUPrimitiveSphere,
                                       SphrSurfaceFromSphr>;

using BasicMatEmptyBatch = GPUMaterialBatch<TracerBasic,
                                            BasicMat,
                                            GPUPrimitiveEmpty,
                                            EmptySurfaceFromEmpty>;

using BasicMatTriBatch = GPUMaterialBatch<TracerBasic,
                                          BasicMat,
                                          GPUPrimitiveTriangle,
                                          EmptySurfaceFromTri>;

using BasicMatSphrBatch = GPUMaterialBatch<TracerBasic,
                                           BasicMat,
                                           GPUPrimitiveSphere,
                                           EmptySurfaceFromSphr>;

using BarycentricMatTriBatch = GPUMaterialBatch<TracerBasic,
                                                BarycentricMat,
                                                GPUPrimitiveTriangle,
                                                BarySurfaceFromTri>;

using SphericalMatSphrBatch = GPUMaterialBatch<TracerBasic,
                                               SphericalMat,
                                               GPUPrimitiveSphere,
                                               SphrSurfaceFromSphr>;