#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "RayStructs.h"
#include "Random.cuh"

#include "RayLib/HemiDistribution.h"
#include "RayLib/MemoryAlignment.h"

#include "GPUPrimitiveP.cuh"
#include "CudaSystem.hpp"

// Meta Primitive Related Light
template <class PGroup>
class GPULight : public GPULightP
{
    public:
        using PData = typename PGroup::PrimitiveData;

    private:        
        const PData&        gPData;
        PrimitiveId         primId;

        static constexpr auto PrimSamplePos    = PGroup::SamplePosition;
        static constexpr auto PrimPdfPos       = PGroup::PdfPosition;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULight(// Common Data
                                         const PData& pData,
                                         PrimitiveId,
                                         // Base Class Related
                                         const TextureRefI<2, Vector3f>& gRad,
                                         uint16_t mediumId,                                         
                                         const GPUTransformI&);
                                ~GPULight() = default;
        // Interface
        __device__ void         Sample(// Output
                                       float& distance,
                                       Vector3& direction,
                                       float& pdf,
                                       // Input
                                       const Vector3& worldLoc,
                                       // I-O
                                       RandomGPU&) const override;

        __device__ void         GenerateRay(// Output
                                            RayReg&,
                                            // Input
                                            const Vector2i& sampleId,
                                            const Vector2i& sampleMax,
                                            // I-O
                                            RandomGPU&,
                                            // Options
                                            bool antiAliasOn = true) const override;

        __device__ float        Pdf(const Vector3& direction,
                                    const Vector3& position) const override;

        __device__ bool         CanBeSampled() const override;
};

template <class PGroup>
class CPULightGroup : public CPULightGroupP<GPULight<PGroup>>
{
    public:
        static constexpr const char*    TypeName() { return PGroup::TypeName(); }

        using PData                     = typename PGroup::PrimitiveData;

    private:
        const PGroup&                   primGroup;
        // Copy of the PData on GPU Memory (only pointer tho)
        const PData*                    dPData;
        // Temp Host Data
        std::vector<PrimitiveId>        hPrimitiveIds;

    protected:
    public:
        // Cosntructors & Destructor
                                    CPULightGroup(const CudaGPU& gpu,
                                                  const GPUPrimitiveGroupI*);
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
                                                       const CudaSystem&) override;

        size_t					    UsedCPUMemory() const override;
};

template <class PGroup>
__device__ GPULight<PGroup>::GPULight(// Common Data
                                      const PData& pData,
                                      PrimitiveId pId,
                                      // Base Class Related
                                      const TextureRefI<2, Vector3f>& gRad,
                                      uint16_t mediumId,
                                      const GPUTransformI& gTrans)
    : GPULightP(gRad, mediumId, gTrans)
    , gPData(gPData)
    , primId(pId)
{}

template <class PGroup>
__device__ void GPULight<PGroup>::Sample(// Output
                                         float& distance,
                                         Vector3& direction,
                                         float& pdf,
                                         // Input
                                         const Vector3& worldLoc,
                                         // I-O
                                         RandomGPU& rng) const
{
    Vector3 normal;
    Vector3 position = PrimSamplePos(normal,
                                     pdf,
                                     //
                                     primId,
                                     gPData,
                                     rng);
    // Transform
    position = gTransform.LocalToWorld(position);
    normal = gTransform.LocalToWorld(normal, true);

    direction = position - worldLoc;
    float distanceSqr = direction.LengthSqr();
    distance = sqrt(distanceSqr);
    direction *= (1.0f / distance);

    //float nDotL = max(normal.Dot(-direction), 0.0f);
    float nDotL = abs(normal.Dot(-direction));
    pdf *= distanceSqr / nDotL;
}

template <class PGroup>
__device__ void  GPULight<PGroup>::GenerateRay(// Output
                                               RayReg& rReg,
                                               // Input
                                               const Vector2i& sampleId,
                                               const Vector2i& sampleMax,
                                               // I-O
                                               RandomGPU& rng,
                                               // Options
                                               bool antiAliasOn) const
{
    // TODO: Add 2D segmentation (Distributed RT)
    float pdf;
    Vector3 normal;
    Vector3 position = PrimSamplePos(normal,
                                     pdf,
                                     //
                                     primId,
                                     gPData,
                                     rng);

    Vector2 xi(GPUDistribution::Uniform<float>(rng),
               GPUDistribution::Uniform<float>(rng));
    Vector3 direction = HemiDistribution::HemiUniformCDF(xi, pdf);
    direction.NormalizeSelf();

    // Generated direction vector is on surface space (hemisperical)
    // Convert it to normal oriented hemisphere (world space)
    QuatF q = Quat::RotationBetweenZAxis(normal);
    direction = q.ApplyRotation(direction);

    // Convert Ray to Worldspace
    position = gTransform.LocalToWorld(position);
    direction = gTransform.LocalToWorld(direction);

    RayF ray = {position, direction};
    rReg = RayReg(ray, 0, INFINITY);
}

template <class PGroup>
__device__ float GPULight<PGroup>::Pdf(const Vector3& direction,
                                       const Vector3& position) const
{    
    // First check if we are actually intersecting
    float distance, pdf;
    Vector3 normal;
    PrimPdfPos(normal,
               pdf,
               distance,
               //
               position,
               direction,
               gTransform,
               primId,
               gPData);

    if(isnan(pdf))
        printf("primPDF NAN\n");

    if(pdf != 0.0f)
    {
        float distanceSqr = distance * distance;
        float nDotL = abs(normal.Dot(-direction));
        pdf *= distanceSqr / nDotL;

        if(isnan(pdf))
            printf("2     primPDF NAN\n");

        return pdf;
    }
    else return 0.0f;
}

template <class PGroup>
__device__ bool GPULight<PGroup>::CanBeSampled() const
{
    return true;
}

template <class PGroup>
CPULightGroup<PGroup>::CPULightGroup(const CudaGPU& gpu, 
                                     const GPUPrimitiveGroupI* pg)
    : CPULightGroupP<GPULight<PGroup>>(gpu)
    , primGroup(static_cast<const PGroup&>(*pg))
    , dPData(nullptr)    
{}

template <class PGroup>
const char* CPULightGroup<PGroup>::Type() const
{
    return Type();
}

template <class PGroup>
size_t CPULightGroup<PGroup>::UsedCPUMemory() const
{
    size_t totalSize = (CPULightGroupP<GPULight<PGroup>>::UsedCPUMemory() +
                        hPrimitiveIds.size());

    return totalSize;
}

#include "GPULightPrimitive.hpp"