#include "LightCameraKernels.h"
#include "GPUCamera.cuh"
#include "GPULight.cuh"
#include "DeviceMemory.h"
#include "CudaConstants.h"
#include "RayLib/SceneStructs.h"

__device__ void AllocateSingleCamera(GPUCameraI*& gPtr,
                                     void* const gMemory,
                                     const CPUCamera& cpuCamera)
{
    gPtr = nullptr;
    switch(cpuCamera.type)
    {
        case CameraType::PINHOLE:
        {
            gPtr = new (gMemory) PinholeCamera(cpuCamera);
            break;
        }
        case CameraType::ORTHOGRAPHIC:
        {
            break;
        }
        case CameraType::APERTURE:
        {
            break;
        }       
        default:
        {
            asm("trap;");
            break;
        }
    }
}

__device__ void AllocateSingleLight(GPULightI*& gPtr,
                                    void* const gMemory,
                                    const CPULight& cpuLight)
{
    gPtr = nullptr;
    switch(cpuLight.type)
    {
        case LightType::POINT:
        {
            gPtr = new (gMemory) PointLight(cpuLight.position0,
                                            cpuLight.flux,
                                            cpuLight.matKey);
            break;
        }
        case LightType::DIRECTIONAL:
        {
            break;
        }
        case LightType::SPOT:      
        {
            break;
        }
        case LightType::RECTANGULAR:
        {
            break;
        }
        case LightType::TRIANGULAR:
        {
            break;
        }
        case LightType::DISK:
        {
            break;
        }
        case LightType::SPHERICAL:
        {
            break;
        }
        default:
        {
            asm("trap;");
            break;
        }
    }
}

__global__ void KCAllocateSingleLight(GPULightI* gPtr,
                                      void* const gMemory,
                                      const CPULight cpuLight)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0)
    {
        AllocateSingleLight(gPtr, gMemory, cpuLight);
    }
}

__global__ void KCAllocateSingleCamera(GPUCameraI*& gPtr,
                                       void* const gMemory,
                                       const CPUCamera cpuCamera)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0)
    {
        AllocateSingleCamera(gPtr, gMemory, cpuCamera);
    }
}


__global__ void KCAllocateLights(GPULightI** gPtrs,
                                 Byte* gMemory,
                                 //
                                 const CPULight* gCPULights,
                                 const uint32_t lightCount,
                                 const uint32_t stride)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < lightCount;
        globalId += blockDim.x * gridDim.x)
    {
        AllocateSingleLight(gPtrs[globalId],
                            gMemory + globalId * stride,
                            gCPULights[globalId]);
    }
}

__global__ void KCAllocateCameras(GPUCameraI** gPtrs,
                                  Byte* const gMemory,
                                  //
                                  const CPUCamera* gCPUCameras,
                                  const uint32_t cameraCount,
                                  const uint32_t stride)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < cameraCount;
        globalId += blockDim.x * gridDim.x)
    {
        AllocateSingleCamera(gPtrs[globalId],
                             gMemory + globalId * stride,
                             gCPUCameras[globalId]);
    }
}

size_t LightCameraKernels::LightClassesUnionSize()
{
    return GPUCameraUnionSize;
}

size_t LightCameraKernels::CameraClassesUnionSize()
{
    return GPULightUnionSize;
}

void LightCameraKernels::ConstructLights(// Output
                                         GPULightI** gPtrs,
                                         Byte* gMemory,
                                         // Input
                                         const std::vector<CPULight>& lightData,
                                         const CudaSystem& system)
{
    // Allocate to GPU Memory
    size_t lightsCPUSize = lightData.size() * sizeof(CPULight);
    DeviceMemory temp(lightsCPUSize);
    CUDA_CHECK(cudaMemcpy(temp, lightData.data(), lightsCPUSize,
                          cudaMemcpyHostToDevice));

    const CudaGPU& gpu = system.BestGPU();
    gpu.AsyncGridStrideKC_X(0, lightData.size(),
                            KCAllocateLights,

                            gPtrs,
                            gMemory,
                            static_cast<const CPULight*>(temp),
                            static_cast<uint32_t>(lightData.size()),
                            static_cast<uint32_t>(LightClassesUnionSize()));
}

void LightCameraKernels::ConstructCameras(// Output
                                          GPUCameraI** gPtrs,
                                          Byte* gMemory,
                                          // Input
                                          const std::vector<CPUCamera>& cameraData,
                                          const CudaSystem& system)
{
    // Befo
     // Allocate to GPU Memory
    size_t camerasCPUSize = cameraData.size() * sizeof(CPUCamera);
    DeviceMemory temp(camerasCPUSize);
    CUDA_CHECK(cudaMemcpy(temp, cameraData.data(), camerasCPUSize,
                          cudaMemcpyHostToDevice));

    const CudaGPU& gpu = system.BestGPU();
    gpu.AsyncGridStrideKC_X(0, cameraData.size(),
                            KCAllocateCameras,

                            gPtrs,
                            gMemory,
                            //
                            static_cast<const CPUCamera*>(temp),
                            static_cast<uint32_t>(cameraData.size()),
                            static_cast<uint32_t>(LightClassesUnionSize()));
}

void LightCameraKernels::ConstructSingleLight(// Output
                                              GPULightI*& gPtr,
                                              Byte* objectMemory,
                                              // Input
                                              const CPULight& lightData,
                                              const CudaSystem& system)
{
    const CudaGPU& gpu = system.BestGPU();
    gpu.KC_X(0, (cudaStream_t)0, 1,
             KCAllocateSingleLight,
             //
             gPtr,
             objectMemory,
             lightData);
}

void LightCameraKernels::ConstructSingleCamera(// Output
                                               GPUCameraI*& gPtr,
                                               Byte* objectMemory,
                                               // Input
                                               const CPUCamera& cameraData,
                                               const CudaSystem& system)
{
    const CudaGPU& gpu = system.BestGPU();
    gpu.KC_X(0, (cudaStream_t)0, 1,
             KCAllocateSingleCamera,
             //
             gPtr,
             objectMemory,
             cameraData);
}