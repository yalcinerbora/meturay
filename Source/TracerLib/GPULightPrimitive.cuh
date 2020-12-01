#pragma once

#include "GPULightI.h"
#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "RayStructs.h"
#include "Random.cuh"

#include "RayLib/HemiDistribution.h"
#include "RayLib/MemoryAlignment.h"

#include "TracerLib/GPUPrimitiveP.cuh"

// Meta Primitive Related Light
template <class PGroup>
class GPULight : public GPULightI
{
    public:
        using PData = typename PGroup::PrimitiveData;
        
    private:        
        PrimitiveId                         primId;
        const GPUTransformI&                transform;
        const PData&                        gPData;
        
        static constexpr auto PrimSample    = PGroup::Sample;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULight(HitKey k,
                                         uint16_t mediumIndex,
                                         PrimitiveId pId,
                                         const GPUTransformI& gTransform,
                                         // Common Data
                                         const PData& pData);
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
                                            RandomGPU&) const override;

        __device__ PrimitiveId  PrimitiveIndex() const override;
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
        GPULightList				    gpuLightList;
        uint32_t                        lightCount;
        
    protected:
    public:
        // Cosntructors & Destructor
                                CPULightGroup(const GPUPrimitiveGroupI*);
                                ~CPULightGroup() = default;


        const char*				Type() const override;
		const GPULightList&		GPULights() const override;
		SceneError				InitializeGroup(const ConstructionDataList& lightNodes,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                const MaterialKeyListing& allMaterialKeys,
												double time,
												const std::string& scenePath) override;
		SceneError				ChangeTime(const NodeListing& lightNodes, double time,
										   const std::string& scenePath) override;
		TracerError				ConstructLights(const CudaSystem&,
                                                const GPUTransformI**) override;
		uint32_t				LightCount() const override;

		size_t					UsedGPUMemory() const override;
        size_t					UsedCPUMemory() const override;
};

template <class PGroup>
__device__ GPULight<PGroup>::GPULight(HitKey k,
                                      uint16_t mediumIndex,
                                      PrimitiveId pId,
                                      const GPUTransformI& gTransform,
                                      // Common Data
                                      const PData& gPData)
    : GPUEndpointI(k, mediumIndex)
    , transform(gTransform)
    , primId(pId)
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
                                  primId,
                                  gPData,
                                  rng);
    // Transform
    position = transform.LocalToWorld(position);
    normal = transform.LocalToWorld(normal, true);

    direction = position - worldLoc;
    float distanceSqr = direction.LengthSqr();
    distance = sqrt(distanceSqr);
    direction *= (1.0f / distance);

    float nDotL = max(normal.Dot(-direction), 0.0f);
    pdf *= distanceSqr / nDotL;

}

template <class PGroup>
__device__ void  GPULight<PGroup>::GenerateRay(// Output
                                               RayReg& rReg,
                                               // Input
                                               const Vector2i& sampleId,
                                               const Vector2i& sampleMax,
                                               // I-O
                                               RandomGPU& rng) const
{
    // TODO: Add 2D segmentation (Distributed RT)
    float pdf;
    Vector3 normal;
    Vector3 position = PrimSample(normal,
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
    position = transform.LocalToWorld(position);
    direction = transform.LocalToWorld(direction);

    RayF ray = {position, direction};
    rReg = RayReg(ray, 0, INFINITY);
}

template <class PGroup>
__device__ PrimitiveId GPULight<PGroup>::PrimitiveIndex() const
{
    return primId;
}

template <class PGroup>
CPULightGroup<PGroup>::CPULightGroup(const GPUPrimitiveGroupI* pg)
    : CPULightGroupI()
    , primGroup(static_cast<const PGroup&>(*pg))
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