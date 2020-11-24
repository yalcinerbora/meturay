#pragma once

#include "GPULightI.cuh"
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