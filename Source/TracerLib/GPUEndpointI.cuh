#pragma once

#include "RayLib/Vector.h"
#include "RayLib/HitStructs.h"

class RandomGPU;
struct RayReg;

// Endpoint interface
// it is either camera or a light source
// Only dynamic polymorphism case for tracer
class GPUEndpointI
{
    protected: 
        // Material of the Light
        // In order to acquire light visibility
        // Launch a ray using this key
        // if nothing hits Tracer batches the ray with this key
        HitKey                  boundaryMaterialKey;
        // Primitive Id is required for primitive lights for NEE estimation
        PrimitiveId             primitiveId;
        
    public:
        __device__              GPUEndpointI(HitKey k, PrimitiveId id);
        virtual                 ~GPUEndpointI() = default;

        // Interface
        // Sample the endpoint from a point
        // Return directional pdf
        // Direction is NOT normalized (and should not be normalized)
        // that data may be usefull (when sampling a light source using NEE)
        // tMax can be generated from it
        virtual __device__ void Sample(// Output
                                       HitKey& materialKey,
                                       PrimitiveId& primId,
                                       //
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

        __device__ HitKey       BoundaryMaterial() const;
        __device__ PrimitiveId  Primitive() const;
};

// Additional to sampling stuff, Light returns flux
// which can be used to determine light importance
class GPULightI : public GPUEndpointI
{
    protected:
        Vector3                         flux;

    public: 
        __device__                      GPULightI(const Vector3& flux, HitKey k, PrimitiveId id);
        virtual                         ~GPULightI() = default;
        // Interface
        virtual __device__ Vector3      Flux(const Vector3& direction) const = 0;
};

__device__      
inline  GPUEndpointI::GPUEndpointI(HitKey k, PrimitiveId id) 
    : boundaryMaterialKey(k) 
    , primitiveId(id)
{}

__device__
inline HitKey GPUEndpointI::BoundaryMaterial() const
{
    return boundaryMaterialKey;
}

__device__
inline PrimitiveId GPUEndpointI::Primitive() const
{
    return primitiveId;
}

__device__
inline GPULightI::GPULightI(const Vector3& flux, HitKey k, PrimitiveId id)
    : GPUEndpointI(k, id)
    , flux(flux)
{}

using GPUCameraI = GPUEndpointI;