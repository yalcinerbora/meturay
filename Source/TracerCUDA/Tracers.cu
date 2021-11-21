#include "Tracers.h"

#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "GPULightSamplerUniform.cuh"
#include "DeviceMemory.h"

template <class T>
__global__ void KCConstructLightSampler(T* loc,
                                        const GPULightI** gLights,
                                        const uint32_t lightCount)
{
    uint32_t globalId = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalId == 0)
    {
        T* lightSampler = new (loc) T(gLights, lightCount);
    }
}

TracerError LightSamplerCommon::ConstructLightSampler(DeviceMemory& memory,
                                                      const GPUDirectLightSamplerI*& dLightSampler,

                                                      LightSamplerType lt,
                                                      const GPULightI** dLights,
                                                      const uint32_t lightCount,

                                                      const CudaSystem& cudaSystem)
{
    switch(lt)
    {
        case LightSamplerType::UNIFORM:
        {
            GPUMemFuncs::EnlargeBuffer(memory, sizeof(GPULightSamplerUniform));
            dLightSampler = static_cast<const GPUDirectLightSamplerI*>(memory);

            const auto& gpu = cudaSystem.BestGPU();
            gpu.KC_X(0, (cudaStream_t)0, 1,
                     // Kernel
                     KCConstructLightSampler<GPULightSamplerUniform>,
                     // Args
                     static_cast<GPULightSamplerUniform*>(memory),
                     dLights,
                     lightCount);

            return TracerError::OK;
        }
        default:
            return TracerError::UNABLE_TO_INITIALIZE_TRACER;
    }
}