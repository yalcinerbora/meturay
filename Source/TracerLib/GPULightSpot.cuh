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

};

//===========//
// GPU POINT //
//===========//
GPULightSpot::GPULightSpot(// Per Light Data
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