#pragma once

#include "Surfaces.h"

#include "TracerLib/GPUMaterialP.cuh"

// Shade Functions
__device__
inline void ConstantBoundaryMatShade(// Output
                                     Vector4* gImage,
                                     // Input as registers
                                     const RayReg& ray,
                                     const RayAuxBasic& aux,
                                     //
                                     RandomGPU& rng,
                                     // Input as global memory
                                     const ConstantBoundaryMatData& gMatData)
{
    Vector3f output = gMatData.backgroundColor * aux.totalRadiance;
    gImage[aux.pixelId][0] = output[0];
    gImage[aux.pixelId][1] = output[1];
    gImage[aux.pixelId][2] = output[2];
}

__device__
inline void BasicMatShade(// Output
                          Vector4f* gImage,
                          //
                          RayGMem* gOutRays,
                          RayAuxBasic* gOutRayAux,
                          const uint32_t maxOutRay,
                          // Input as registers
                          const RayReg& ray,
                          const EmptySurface& surface,
                          const RayAuxBasic& aux,
                          //
                          RandomGPU& rng,
                          // Input as global memory
                          const ConstantAlbedoMatData& gMatData,
                          const HitKey::Type& matId)
{
    gImage[aux.pixelId][0] = gMatData.dAlbedo[matId][0];
    gImage[aux.pixelId][1] = gMatData.dAlbedo[matId][1];
    gImage[aux.pixelId][2] = gMatData.dAlbedo[matId][2];
}

__device__
inline void BaryMatShade(// Output
                         Vector4f* gImage,
                         //
                         RayGMem* gOutRays,
                         RayAuxBasic* gOutRayAux,
                         const uint32_t maxOutRay,
                         // Input as registers
                         const RayReg& ray,
                         const BarySurface& surface,
                         const RayAuxBasic& aux,
                         //
                         RandomGPU& rng,
                         // Input as global memory
                         const ConstantAlbedoMatData& gMatData,
                         const HitKey::Type& matId)
{
    gImage[aux.pixelId][0] = surface.baryCoords[0];
    gImage[aux.pixelId][1] = surface.baryCoords[1];
    gImage[aux.pixelId][2] = surface.baryCoords[2];
}

// Material Groups
class ConstantBoundaryMat final
    : public GPUBoundaryMatGroup<TracerBasic,
                                 ConstantBoundaryMatData,
                                 ConstantBoundaryMatShade>
{
    public:
        static constexpr const char*    TypeName() { return "ConstantBoundary"; }

    private:
        DeviceMemory                        memory;

        // CPU
        std::map<uint32_t, uint32_t>        innerIds;

    public:
        // Constructors & Destructor
                                    ConstantBoundaryMat(int gpuId);
                                    ~ConstantBoundaryMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*                 Type() const override;
        // Allocates and Generates Data
        SceneError                  InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time) override;
        SceneError                  ChangeTime(const std::set<SceneFileNode>& materialNodes, double time) override;

        // Material Queries
        int                         InnerId(uint32_t materialId) const override;
        bool                            IsLoaded(uint32_t materialId) const override;

        size_t                      UsedGPUMemory() const override;
        size_t                      UsedCPUMemory() const override;

        size_t                      UsedGPUMemory(uint32_t materialId) const override;
        size_t                      UsedCPUMemory(uint32_t materialId) const override;

        uint8_t                     OutRayCount() const override;
};

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

    protected:
    public:
                                        BasicMat(int gpuId);
                                        ~BasicMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*                     Type() const override;
        // Allocates and Generates Data
        SceneError                      InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time) override;
        SceneError                      ChangeTime(const std::set<SceneFileNode>& materialNodes, double time) override;

        // Material Queries
        int                             InnerId(uint32_t materialId) const override;
        bool                            IsLoaded(uint32_t materialId) const override;

        size_t                          UsedGPUMemory() const override;
        size_t                          UsedCPUMemory() const override;

        size_t                          UsedGPUMemory(uint32_t materialId) const override;
        size_t                          UsedCPUMemory(uint32_t materialId) const override;

        uint8_t                         OutRayCount() const override;
};

class BarycentricMat final
    : public GPUMaterialGroup<TracerBasic,
                              ConstantAlbedoMatData,
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
        const char*                     Type() const override;
        // Allocates and Generates Data
        SceneError                      InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time) override;
        SceneError                      ChangeTime(const std::set<SceneFileNode>& materialNodes, double time) override;

        // Material Queries
        int                             InnerId(uint32_t materialId) const override;
        bool                            IsLoaded(uint32_t materialId) const override;

        size_t                          UsedGPUMemory() const override;
        size_t                          UsedCPUMemory() const override;

        size_t                          UsedGPUMemory(uint32_t materialId) const override;
        size_t                          UsedCPUMemory(uint32_t materialId) const override;

        uint8_t                         OutRayCount() const override;
};

// Material Batches
extern template class GPUBoundaryMatBatch<TracerBasic, 
                                          ConstantBoundaryMat>;

extern template class GPUMaterialBatch<TracerBasic,
                                       BarycentricMat,
                                       GPUPrimitiveTriangle,
                                       BarySurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
                                       BasicMat,
                                       GPUPrimitiveTriangle,
                                       EmptySurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
                                       BasicMat,
                                       GPUPrimitiveSphere,
                                       EmptySurfaceFromSphr>;

using ConstantBoundaryMatBatch = GPUBoundaryMatBatch<TracerBasic, 
                                                     ConstantBoundaryMat>;

using BarycentricMatTriBatch = GPUMaterialBatch<TracerBasic,
                                                BarycentricMat,
                                                GPUPrimitiveTriangle,
                                                BarySurfaceFromTri>;

using BasicMatTriBatch = GPUMaterialBatch<TracerBasic,
                                          BasicMat,
                                          GPUPrimitiveTriangle,
                                          EmptySurfaceFromTri>;

using BasicMatSphrBatch = GPUMaterialBatch<TracerBasic,
                                           BasicMat,
                                           GPUPrimitiveSphere,
                                           EmptySurfaceFromSphr>;