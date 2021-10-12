#pragma once

#include "GPUEndpointI.h"

// Do not delete this file
// maybe these classes will be required to be implmented

struct UVSurface;

class GPULightI : public GPUEndpointI
{   
    public:
        __device__                      GPULightI(uint16_t mediumIndex,
                                                  const GPUTransformI&);
        virtual                         ~GPULightI() = default;
        
        // Interface
        virtual __device__ Vector3f     Emit(const Vector3& wo,
                                             const Vector3& pos,
                                             //
                                             const UVSurface&) = 0;
};
    
class CPULightGroupI : public CPUEndpointGroupI {};

__device__
inline GPULightI::GPULightI(uint16_t mediumIndex,
                            const GPUTransformI& gTrans)
    : GPUEndpointI(mediumIndex, gTrans)
{}