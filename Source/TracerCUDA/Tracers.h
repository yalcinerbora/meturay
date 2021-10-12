#pragma once

#include <string>

#include "RayLib/TracerError.h"

class GPUDirectLightSamplerI;
class DeviceMemory;
class CudaSystem;

class GPULightI;

enum class LightSamplerType
{
    UNIFORM,

    END
};

namespace LightSamplerCommon
{
    TracerError             StringToLightSamplerType(LightSamplerType&,
                                                     const std::string&);
    std::string             LightSamplerTypeToString(LightSamplerType);
    TracerError             ConstructLightSampler(DeviceMemory& memory,
                                                  const GPUDirectLightSamplerI*& lightSampler,

                                                  LightSamplerType lt,
                                                  const GPULightI** dLights,
                                                  const uint32_t lightCount,

                                                  const CudaSystem& cudaSystem);
}