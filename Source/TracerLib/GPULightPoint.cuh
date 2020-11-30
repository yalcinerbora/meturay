#pragma once

#include "GPULightI.h"
#include "GPUTransformI.h"
#include "DeviceMemory.h"

class GPULightPoint : public GPULightI
{
    private:        
        Vector3f            position;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightPoint(// Per Light Data
                                              TransformId tIndex,
                                              const Vector3f& position,
                                              // Common Data
                                              const GPUTransformI** gTransforms,
                                              // Endpoint Related Data
                                              HitKey k, uint16_t mediumIndex);
                                ~GPULightPoint() = default;
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

class CPULightGroupPoint : public CPULightGroupI
{
    private:
    protected:
    public:
        // Cosntructors & Destructor
                                CPULightGroupPoint(const GPUPrimitiveGroupI*);
                                ~CPULightGroupPoint() = default;


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

        void                    AttachGlobalTransformArray(const GPUTransformI** deviceTranfsorms) override;
};

GPULightPoint::GPULightPoint(// Per Light Data
                             TransformId tIndex,
                             const Vector3f& position,
                             // Common Data
                             const GPUTransformI** gTransforms,
                             // Endpoint Related Data
                             HitKey k, uint16_t mediumIndex)
    : GPUEndpointI(k, mediumIndex)
    , position(gTransforms[tIndex]->LocalToWorld(position))
{}

__device__ void GPULightPoint::Sample(// Output
                                      float& distance,
                                      Vector3& direction,
                                      float& pdf,
                                      // Input
                                      const Vector3& worldLoc,
                                      // I-O
                                      RandomGPU&) const
{
    direction = (position - worldLoc);
    distance = direction.Length();
    direction *= (1.0f / distance);

    // Fake pdf to incorporate square faloff
    pdf = (distance * distance);
}

__device__ void GPULightPoint::GenerateRay(// Output
                                           RayReg&,
                                           // Input
                                           const Vector2i& sampleId,
                                           const Vector2i& sampleMax,
                                           // I-O
                                           RandomGPU&) const
{
    // TODO: Implement
}

__device__ PrimitiveId GPULightPoint::PrimitiveIndex() const
{
    return 0;
}