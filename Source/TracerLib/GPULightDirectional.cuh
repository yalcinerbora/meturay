#pragma once

#include "GPULightI.h"
#include "GPUTransformI.h"
#include "DeviceMemory.h"

class GPULightDirectional : public GPULightI
{
    private:        
        Vector3f                direction;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightDirectional(// Per Light Data
                                                    TransformId tIndex,
                                                    const Vector3f& direction,
                                                    // Common Data
                                                    const GPUTransformI** gTransforms,
                                                    // Endpoint Related Data
                                                    HitKey k, uint16_t mediumIndex);
                                ~GPULightDirectional() = default;
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


class CPULightGroupDirectional : public CPULightGroupI
{
    public:
        // Constructors & Destructor
                                CPULightGroupDirectional(const GPUPrimitiveGroupI*);
                                ~CPULightGroupDirectional() = default;


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

GPULightDirectional::GPULightDirectional(// Per Light Data
                                         TransformId tIndex,
                                         const Vector3f& direction,
                                         // Common Data
                                         const GPUTransformI** gTransforms,
                                         // Endpoint Related Data
                                         HitKey k, uint16_t mediumIndex)
    : GPUEndpointI(k, mediumIndex)
    , direction(gTransforms[tIndex]->LocalToWorld(direction))
{}

__device__ void GPULightDirectional::Sample(// Output
                                            float& distance,
                                            Vector3& dir,
                                            float& pdf,
                                            // Input
                                            const Vector3& worldLoc,
                                            // I-O
                                            RandomGPU&) const
{
    dir = -direction;
    distance = FLT_MAX;
    pdf = 1.0f;
}

__device__ void GPULightDirectional::GenerateRay(// Output
                                                 RayReg&,
                                                 // Input
                                                 const Vector2i& sampleId,
                                                 const Vector2i& sampleMax,
                                                 // I-O
                                                 RandomGPU& rng) const
{
    // TODO: implement
}

__device__ PrimitiveId GPULightDirectional::PrimitiveIndex() const
{
    return 0;
}