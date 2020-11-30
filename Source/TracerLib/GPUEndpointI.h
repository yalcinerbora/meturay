#pragma once

#include "RayLib/Vector.h"
#include "RayLib/HitStructs.h"

class GPUDistribution2D;
class RandomGPU;
struct RayReg;

// Endpoint interface
// it is either camera or a light source
// Only dynamic polymorphism case for tracer
class GPUEndpointI
{
    protected: 
        // Material of the Endpoint
        // In order to acquire light visibility
        // Launch a ray using this key
        // if nothing hits Tracer batches the ray with this key
        HitKey                  boundaryMaterialKey;
        // Medium of the endpoint
        // used to initialize rays when generated
        uint16_t                mediumIndex;
        
    public:
        __device__              GPUEndpointI(HitKey k, uint16_t mediumIndex);
        virtual                 ~GPUEndpointI() = default;

        // Interface
        // Sample the endpoint from a point
        // Return directional pdf
        // Direction is NOT normalized (and should not be normalized)
        // that data may be usefull (when sampling a light source using NEE)
        // tMax can be generated from it
        virtual __device__ void Sample(// Output                                       
                                       float& distance,
                                       Vector3& direction,
                                       float& pdf,
                                       // Input
                                       const Vector3& position,
                                       // I-O
                                       RandomGPU&) const = 0;

        // Generate a Ray from this endpoint
        virtual __device__ void GenerateRay(// Output
                                            RayReg&,
                                            // Input
                                            const Vector2i& sampleId,
                                            const Vector2i& sampleMax,
                                            // I-O
                                            RandomGPU&) const = 0;

        virtual __device__ PrimitiveId  PrimitiveIndex() const = 0;

        __device__ HitKey               BoundaryMaterial() const;
        __device__ uint16_t             MediumIndex() const;
        //__device__ TransformId          TransformIndex() const;
};

__device__      
inline  GPUEndpointI::GPUEndpointI(HitKey k, uint16_t mediumIndex) 
    : boundaryMaterialKey(k) 
    , mediumIndex(mediumIndex)
{}

__device__
inline HitKey GPUEndpointI::BoundaryMaterial() const
{
    return boundaryMaterialKey;
}

__device__
inline uint16_t GPUEndpointI::MediumIndex() const
{
    return mediumIndex;
}