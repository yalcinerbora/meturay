#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "RayStructs.h"
#include "RNGenerator.h"
#include "MangledNames.h"

#include "RayLib/HemiDistribution.h"
#include "RayLib/MemoryAlignment.h"

#include "GPUPrimitiveP.cuh"
#include "CudaSystem.hpp"
#include "GPUSurface.h"

// Meta Primitive Related Light
template <class PGroup>
class GPULight final : public GPULightP
{
    public:
        using PData = typename PGroup::PrimitiveData;

    private:
        const PData&        gPData;
        PrimitiveId         primId;

        static constexpr auto PrimSamplePos     = PGroup::SamplePosition;
        static constexpr auto PrimPosPdfRef     = PGroup::PositionPdfRef;
        static constexpr auto PrimPosPdfHit     = PGroup::PositionPdfHit;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULight(// Common Data
                                         const PData& gPData,
                                         PrimitiveId,
                                         // Base Class Related
                                         const TextureRefI<2, Vector3f>& gRad,
                                         uint16_t mediumId, HitKey,
                                         const GPUTransformI&);
                                ~GPULight() = default;
        // Interface
        __device__ void         Sample(// Output
                                       float& distance,
                                       Vector3& direction,
                                       float& pdf,
                                       Vector2f& localCoords,
                                       // Input
                                       const Vector3& worldLoc,
                                       // I-O
                                       RNGeneratorGPUI&) const override;

        __device__ void         GenerateRay(// Output
                                            RayReg&,
                                            Vector2f& localCoords,
                                            // Input
                                            const Vector2i& sampleId,
                                            const Vector2i& sampleMax,
                                            // I-O
                                            RNGeneratorGPUI&,
                                            // Options
                                            bool antiAliasOn = true) const override;

        __device__ float        Pdf(const Vector3& direction,
                                    const Vector3& position) const override;
        __device__ float        Pdf(float distance,
                                    const Vector3& hitPosition,
                                    const Vector3& direction,
                                    const QuatF& tbnRotation) const override;

        // Photon Stuff
        __device__ Vector3f     GeneratePhoton(// Output
                                               RayReg& rayOut,
                                               Vector3f& normal,
                                               float& posPDF,
                                               float& dirPDF,
                                               // I-O
                                               RNGeneratorGPUI&) const override;

        __device__ bool         CanBeSampled() const override;
        __device__ bool         IsPrimitiveBackedLight() const override;
};

template <class PGroup>
class CPULightGroup final : public CPULightGroupP<GPULight<PGroup>, PGroup>
{
    public:
        TYPENAME_DEF(LightGroup, PGroup::TypeName());

        using Base                      = CPULightGroupP<GPULight<PGroup>, PGroup>;
        using PrimitiveData             = typename Base::PrimitiveData;

    private:
        const PGroup&                   primGroup;
        // Copy of the PData on GPU Memory (it contains only pointers)
        const PrimitiveData*            dPData;
        // Temp Host Data
        std::vector<PrimitiveId>        hPrimitiveIds;
        std::vector<HitKey>             hPackedWorkKeys;

    protected:
    public:
        // Constructors & Destructor
                                    CPULightGroup(const GPUPrimitiveGroupI*,
                                                  const CudaGPU& gpu);
                                    ~CPULightGroup() = default;

        const char*				    Type() const override;
        SceneError				    InitializeGroup(const EndpointGroupDataList& lightNodes,
                                                    const TextureNodeMap& textures,
                                                    const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                    const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                    uint32_t batchId, double time,
                                                    const std::string& scenePath) override;
        SceneError				    ChangeTime(const NodeListing& lightNodes, double time,
                                               const std::string& scenePath) override;
        TracerError				    ConstructEndpoints(const GPUTransformI**,
                                                       const AABB3f&,
                                                       const CudaSystem&) override;

        const std::vector<HitKey>&  PackedHitKeys() const override;
        size_t					    UsedCPUMemory() const override;
};

template <class PGroup>
__device__ GPULight<PGroup>::GPULight(// Common Data
                                      const PData& gPData,
                                      PrimitiveId pId,
                                      // Base Class Related
                                      const TextureRefI<2, Vector3f>& gRad,
                                      uint16_t mediumId, HitKey hk,
                                      const GPUTransformI& gTrans)
    : GPULightP(gRad, mediumId, hk, gTrans)
    , gPData(gPData)
    , primId(pId)
{}

template <class PGroup>
__device__ void GPULight<PGroup>::Sample(// Output
                                         float& distance,
                                         Vector3& direction,
                                         float& pdf,
                                         Vector2f& localCoords,
                                         // Input
                                         const Vector3& worldLoc,
                                         // I-O
                                         RNGeneratorGPUI& rng) const
{
    Vector3 normal;
    Vector3 position = PrimSamplePos(normal,
                                     pdf,
                                     //
                                     gTransform,
                                     //
                                     primId,
                                     gPData,
                                     rng);

    direction = position - worldLoc;
    float distanceSqr = direction.LengthSqr();
    distance = sqrt(distanceSqr);
    direction *= (1.0f / distance);

    float nDotL = abs(normal.Dot(-direction));
    pdf *= distanceSqr / nDotL;

    // TODO: Do some localCoord Generation
    localCoords = Vector2f(NAN, NAN);
}

template <class PGroup>
__device__ void  GPULight<PGroup>::GenerateRay(// Output
                                               RayReg& rReg,
                                               Vector2f& localCoords,
                                               // Input
                                               const Vector2i& sampleId,
                                               const Vector2i& sampleMax,
                                               // I-O
                                               RNGeneratorGPUI& rng,
                                               // Options
                                               bool antiAliasOn) const
{
    // TODO: Add 2D segmentation (Distributed RT)
    float pdf;
    Vector3 normal;
    Vector3 position = PrimSamplePos(normal,
                                     pdf,
                                     //
                                     gTransform,
                                     //
                                     primId,
                                     gPData,
                                     rng);

    Vector2 xi(rng.Uniform(), rng.Uniform());
    Vector3 direction = HemiDistribution::HemiUniformCDF(xi, pdf);
    direction.NormalizeSelf();

    // Generated direction vector is on surface space (hemispherical)
    // Convert it to normal oriented hemisphere (world space)
    QuatF q = Quat::RotationBetweenZAxis(normal);
    direction = q.ApplyRotation(direction);

    RayF ray = {position, direction};
    rReg = RayReg(ray, 0, INFINITY);

    // TODO:
    localCoords = Vector2f(NAN, NAN);
}

template <class PGroup>
__device__ float GPULight<PGroup>::Pdf(const Vector3& direction,
                                       const Vector3& position) const
{
    // First check if we are actually intersecting
    float distance, pdf;
    Vector3 normal;
    PrimPosPdfRef(normal,
                  pdf,
                  distance,
                  //
                  RayF(direction, position),
                  gTransform,
                  primId,
                  gPData);

    return pdf;

    if(isnan(pdf)) printf("primPDF NAN\n");

    if(pdf != 0.0f)
    {
        float distanceSqr = distance * distance;
        float nDotL = abs(normal.Dot(-direction));
        pdf *= distanceSqr / nDotL;

        if(isnan(pdf)) printf("2     primPDF NAN\n");

        return pdf;
    }
    else return 0.0f;
}

template <class PGroup>
__device__ float GPULight<PGroup>::Pdf(float distance,
                                       const Vector3& hitPosition,
                                       const Vector3& direction,
                                       const QuatF& tbnRotation) const
{
    float pdf = PrimPosPdfHit(hitPosition, direction,
                              tbnRotation, gTransform,
                              primId, gPData);
    Vector3f normal = GPUSurface::NormalToSpace(tbnRotation);
    float nDotL = abs(normal.Dot(-direction));
    return pdf * distance * distance / nDotL;
}

template <class PGroup>
__device__
inline Vector3f GPULight<PGroup>::GeneratePhoton(// Output
                                                 RayReg& rayOut,
                                                 Vector3f& normal,
                                                 float& posPDF,
                                                 float& dirPDF,
                                                 // I-O
                                                 RNGeneratorGPUI& rng) const
{
    // TODO: Implement
    return Zero3f;
}

template <class PGroup>
__device__ bool GPULight<PGroup>::CanBeSampled() const
{
    return true;
}

template <class PGroup>
__device__ bool GPULight<PGroup>::IsPrimitiveBackedLight() const
{
    return true;
}

template <class PGroup>
CPULightGroup<PGroup>::CPULightGroup(const GPUPrimitiveGroupI* pg,
                                     const CudaGPU& gpu)
    : Base(*pg, gpu)
    , primGroup(static_cast<const PGroup&>(*pg))
    , dPData(nullptr)
{}

template <class PGroup>
const char* CPULightGroup<PGroup>::Type() const
{
    return TypeName();
}

template <class PGroup>
size_t CPULightGroup<PGroup>::UsedCPUMemory() const
{
    size_t totalSize = (Base::UsedCPUMemory() + hPrimitiveIds.size());

    return totalSize;
}

template <class PGroup>
const std::vector<HitKey>& CPULightGroup<PGroup>::PackedHitKeys() const
{
    return hPackedWorkKeys;
}

#include "GPULightPrimitive.hpp"