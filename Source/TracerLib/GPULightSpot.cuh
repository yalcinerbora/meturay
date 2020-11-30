#pragma once

#include "GPULightI.h"
#include "GPUTransformI.h"
#include "DeviceMemory.h"

class GPULightSpot : public GPULightI
{
    private:        
        Vector3f            position;
        float               cosMin;
        Vector3             direction;
        float               cosMax;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightSpot(// Per Light Data
                                             TransformId tIndex,
                                             const Vector3f& position,
                                             float cosMin,
                                             const Vector3f& direction,
                                             float cosMax,
                                             // Common Data
                                             const GPUTransformI** gTransforms,
                                             // Endpoint Related Data
                                             HitKey k, uint16_t mediumIndex);
                                ~GPULightSpot() = default;
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

class CPULightGroupSpot : public CPULightGroupI
{
    public:
        static constexpr const char*    TypeName(){return "Spot"; }

    private:
        DeviceMemory                    memory;
        //
        std::vector<Vector3f>           hPositions;
        std::vector<Vector3f>           hDirections;
        std::vector<Vector2f>           hCosines;

        std::vector<HitKey>             hHitKeys;
        std::vector<uint16_t>           hMediumIds;
        std::vector<PrimitiveId>        hPrimitiveIds;
        std::vector<TransformId>        hTransformIds;

        // Allocations of the GPU Class
        const GPULightSpot*             dGPULights;
        // GPU pointers to those allocated classes on the CPU
        GPULightList				    gpuLightList;
        uint32_t                        lightCount;

    protected:
    public:
        // Cosntructors & Destructor
                                CPULightGroupSpot(const GPUPrimitiveGroupI*);
                                ~CPULightGroupSpot() = default;


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
		TracerError				ConstructLights(const CudaSystem&) override;
		uint32_t				LightCount() const override;

		size_t					UsedGPUMemory() const override;
		size_t					UsedCPUMemory() const override;
};

__device__
inline GPULightSpot::GPULightSpot(// Per Light Data
                                  TransformId tIndex,
                                  const Vector3f& position,
                                  float cosMin,
                                  const Vector3f& direction,
                                  float cosMax,
                                  // Common Data
                                  const GPUTransformI** gTransforms,
                                  // Endpoint Related Data
                                  HitKey k, uint16_t mediumIndex)
    : GPUEndpointI(k, mediumIndex)
    , position(gTransforms[tIndex]->LocalToWorld(position))
    , direction(gTransforms[tIndex]->LocalToWorld(direction))
    , cosMin(cosMin)
    , cosMax(cosMax)
{}

__device__ void GPULightSpot::Sample(// Output
                                     float& distance,
                                     Vector3& dir,
                                     float& pdf,
                                     // Input
                                     const Vector3& worldLoc,
                                     // I-O
                                     RandomGPU&) const
{
    dir = -direction;
    distance = (position - worldLoc).Length();
    
    // Fake pdf to incorporate square faloff
    pdf = (distance * distance);
}

__device__ void GPULightSpot::GenerateRay(// Output
                                          RayReg&,
                                          // Input
                                          const Vector2i& sampleId,
                                          const Vector2i& sampleMax,
                                          // I-O
                                          RandomGPU&) const
{
    // TODO: Implement
}

__device__ PrimitiveId GPULightSpot::PrimitiveIndex() const
{
    return 0;
}

inline CPULightGroupSpot::CPULightGroupSpot(const GPUPrimitiveGroupI*)
    : CPULightGroupI()
    , lightCount(0)
    , dGPULights(nullptr)
{}

inline const char* CPULightGroupSpot::Type() const
{
    return TypeName();
}

inline const GPULightList& CPULightGroupSpot::GPULights() const
{
    return gpuLightList;
}

inline uint32_t CPULightGroupSpot::LightCount() const
{
    return lightCount;
}

inline size_t CPULightGroupSpot::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPULightGroupSpot::UsedCPUMemory() const
{
    size_t totalSize = (hHitKeys.size() * sizeof(HitKey) +
                        hMediumIds.size() * sizeof(uint16_t) +
                        hPrimitiveIds.size() * sizeof(PrimitiveId) +
                        hTransformIds.size() * sizeof(TransformId) + 
                        hPositions.size() * sizeof(Vector3f) + 
                        hDirections.size() * sizeof(Vector3f) +
                        hCosines.size() * sizeof(Vector2f));

    return totalSize;
}