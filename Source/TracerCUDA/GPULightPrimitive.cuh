#pragma once

#include "GPULightI.h"
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
class GPULight : public GPULightI
{
    public:
        using PData = typename PGroup::PrimitiveData;

    private:        
        const PData&                        gPData;

        static constexpr auto PrimSample    = PGroup::Sample;
        static constexpr auto PrimPdf       = PGroup::Pdf;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULight(// Common Data
                                         const PData& pData,
                                         // Base Class Related
                                         uint16_t mediumId,
                                         HitKey, TransformId,
                                         const GPUTransformI&,
                                         PrimitiveId = 0);
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
class CPULightGroup : public CPULightGroupI
{
    public:
        static constexpr const char*    TypeName() { return PGroup::TypeName(); }

        using PData                     = typename PGroup::PrimitiveData;

    private:
        const PGroup&                   primGroup;
        DeviceMemory                    memory;
        //
        std::vector<HitKey>             hHitKeys;
        std::vector<uint16_t>           hMediumIds;
        std::vector<PrimitiveId>        hPrimitiveIds;
        std::vector<TransformId>        hTransformIds;
        // Copy of the PData on GPU Memory
        const PData*                    dPData;
        // Allocations of the GPU Class
        const GPULight<PGroup>*         dGPULights;
        // GPU pointers to those allocated classes on the CPU
        GPULightList				        gpuLightList;
        uint32_t                        lightCount;

    protected:
    public:
        // Cosntructors & Destructor
                                    CPULightGroup(const GPUPrimitiveGroupI*);
                                    ~CPULightGroup() = default;

        const char*				    Type() const override;
		const GPULightList&		    GPULights() const override;
        SceneError				    InitializeGroup(const LightGroupDataList& lightNodes,
                                                    const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                    const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                    const MaterialKeyListing& allMaterialKeys,
                                                    double time,
                                                    const std::string& scenePath) override;
        SceneError				    ChangeTime(const NodeListing& lightNodes, double time,
                                               const std::string& scenePath) override;
		TracerError				    ConstructLights(const CudaSystem&,
                                                    const GPUTransformI**,
                                                    const KeyMaterialMap&) override;
		uint32_t				        LightCount() const override;

		size_t					    UsedGPUMemory() const override;
        size_t					    UsedCPUMemory() const override;
};

template <class PGroup>
__device__ GPULight<PGroup>::GPULight(// Common Data
                                      const PData& pData,
                                      // Base Class Related
                                      uint16_t mediumId,
                                      HitKey hK, TransformId tId,
                                      const GPUTransformI& gTrans,
                                      PrimitiveId pId)
    : GPULightI(mediumId, hK, 
                tId, pId, gTrans)    
    , gPData(gPData)
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
    Vector3 position = PrimSample(normal,
                                  pdf,
                                  //
                                  primitiveId,
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
    Vector3 position = PrimSample(normal,
                                  pdf,
                                  //
                                  primitiveId,
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
    PrimPdf(normal, 
            pdf,
            distance,            
            //
            position,
            direction,
            gTransform,
            primitiveId, 
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
CPULightGroup<PGroup>::CPULightGroup(const GPUPrimitiveGroupI* pg)
    : primGroup(static_cast<const PGroup&>(*pg))
    , dPData(nullptr)
    , dGPULights(nullptr)
    , lightCount(0)
{}

template <class PGroup>
const char* CPULightGroup<PGroup>::Type() const
{
    return Type();
}

template <class PGroup>
const GPULightList& CPULightGroup<PGroup>::GPULights() const
{
    return gpuLightList;
}

template <class PGroup>
uint32_t  CPULightGroup<PGroup>::LightCount() const
{
    return lightCount;
}

template <class PGroup>
size_t CPULightGroup<PGroup>::UsedGPUMemory() const
{
    return memory.Size();
}

template <class PGroup>
size_t CPULightGroup<PGroup>::UsedCPUMemory() const
{
    size_t totalSize = (hHitKeys.size() +
                        hMediumIds.size() +
                        hPrimitiveIds.size() +
                        hTransformIds.size());

    return totalSize;
}

#include "GPULightPrimitive.hpp"