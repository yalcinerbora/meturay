#pragma once

#include "RayLib/Vector.h"
#include "RayLib/HitStructs.h"

#include "GPULocation.h"

class RandomGPU;
struct RayReg;

// Endpoint interface
// it is either camera or a light source
// Only dynamic polymorphism case for tracer
class GPUEndpointI : public GPULocation
{
    protected:
        // Medium of the endpoint
        // used to initialize rays when generated
        uint16_t                    mediumIndex;

    public:
        __device__                  GPUEndpointI(uint16_t mediumIndex,
                                                 HitKey, TransformId, 
                                                 PrimitiveId,
                                                 const GPUTransformI&);
        virtual                     ~GPUEndpointI() = default;

        // Interface
        // Sample the endpoint from a point
        // Return directional pdf
        // Direction is NOT normalized (and should not be normalized)
        // that data may be usefull (when sampling a light source using NEE)
        // tMax can be generated from it
        virtual __device__ void     Sample(// Output
                                           float& distance,
                                           Vector3& direction,
                                           float& pdf,
                                           // Input
                                           const Vector3& position,
                                           // I-O
                                           RandomGPU&) const = 0;
        // Generate a Ray from this endpoint
        virtual __device__ void     GenerateRay(// Output
                                                RayReg&,
                                                // Input
                                                const Vector2i& sampleId,
                                                const Vector2i& sampleMax,
                                                // I-O
                                                RandomGPU&,
                                                // Options
                                                bool antiAliasOn = true) const = 0;
        virtual __device__ float    Pdf(const Vector3& direction,
                                        const Vector3& position) const = 0;
        virtual __device__ bool     CanBeSampled() const = 0;

        __device__ uint16_t         MediumIndex() const;
};

__device__
inline  GPUEndpointI::GPUEndpointI(uint16_t mediumIndex,
                                   HitKey mK, TransformId tId, PrimitiveId pId,
                                   const GPUTransformI& gTrans)
    : GPULocation(mK, tId, pId, gTrans)
    , mediumIndex(mediumIndex)
{}

__device__
inline uint16_t GPUEndpointI::MediumIndex() const
{
    return mediumIndex;
}