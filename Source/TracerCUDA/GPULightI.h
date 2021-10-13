#pragma once

#include "GPUEndpointI.h"
#include "GPUPrimitiveEmpty.h"

// Do not delete this file
// maybe these classes will be required to be implmented

struct UVSurface;
class GPULightI;

using GPULightList = std::vector<const GPULightI*>;

class GPULightI : public GPUEndpointI
{
    public:
        // Constructors & Destructor
        __device__                      GPULightI(uint16_t mediumIndex,
                                                  HitKey, const GPUTransformI&);
        virtual                         ~GPULightI() = default;

        // Interface
        virtual __device__ Vector3f     Emit(const Vector3& wo,
                                             const Vector3& pos,
                                             //
                                             const UVSurface&) const = 0;
};

class CPULightGroupI : public CPUEndpointGroupI
{
    public:
        virtual                         ~CPULightGroupI() = default;
        // Interface
        virtual const GPULightList&     GPULights() const = 0;
        virtual const CudaGPU&          GPU() const = 0;
};

__device__
inline GPULightI::GPULightI(uint16_t mediumIndex,
                            HitKey hk,
                            const GPUTransformI& gTrans)
    : GPUEndpointI(mediumIndex, hk, gTrans)
{}