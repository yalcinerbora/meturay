#pragma once

#include "GPUEndpointI.h"

// Do not delete this file
// maybe these classes will be required to be implmented

class GPULightI : public GPUEndpointI
{   
    public:
        __device__                      GPULightI(uint16_t mediumIndex,
                                                  const GPUTransformI&);
        virtual                         ~GPULightI() = default;
        
        // Interface
        virtual __device__ Vector3f     Emittance(const Vector3& wo,
                                                  const Vector3& pos,
                                                  //
                                                  const Vector3f& normal,
                                                  const Vector2f& uv) = 0;
};
    
class CPULightGroupI : public CPUEndpointGroupI {};

__device__
inline GPULightI::GPULightI(uint16_t mediumIndex,
                            const GPUTransformI& gTrans)
    : GPUEndpointI(mediumIndex, gTrans)
{}