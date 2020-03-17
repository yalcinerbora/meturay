#pragma once

#include <vector>
#include "RayLib/Types.h"

#include "GPUCamera.cuh"
#include "GPULight.cuh"

#include "DeviceMemory.h"
#include "CudaConstants.h"
#include "RayLib/SceneStructs.h"

#include "Random.cuh"

namespace LightCameraKernels
{
    size_t      LightClassesUnionSize();
    size_t      CameraClassesUnionSize();

    void        ConstructLights(// Output
                                GPULightI** gPtrs,
                                Byte* gMemory,
                                // Input
                                const std::vector<CPULight>& lightData,
                                const CudaSystem&);
    void        ConstructCameras(// Output
                                 GPUCameraI** gPtrs,
                                 Byte* gMemory,
                                 // Input
                                 const std::vector<CPUCamera>& cameraData,
                                 const CudaSystem&);

    void        ConstructSingleLight(// Output
                                     GPULightI*& gPtr,
                                     Byte* gMemory,
                                     // Input
                                     const CPULight& lightData,
                                     const CudaSystem&);
    void        ConstructSingleCamera(// Output
                                      GPUCameraI*& gPtr,
                                      Byte* gMemory,
                                      // Input
                                      const CPUCamera& cameraData,
                                      const CudaSystem&);
};

inline __device__ void AllocateSingleCamera(GPUCameraI*& gPtr,
                                            void* const gMemory,
                                            const CPUCamera& cpuCamera)
{
    gPtr = nullptr;
    switch(cpuCamera.type)
    {
        case CameraType::PINHOLE:
        {
            gPtr = new (gMemory) PinholeCamera(cpuCamera);
            printf("gPtr %p, gMemory %p\n", gPtr, gMemory);

            //curandStateMRG32k3a_t randomMemoryFake;

            //RandomGPU rng(&randomMemoryFake);
            //RayReg r;
            //gPtr->GenerateRay(r, {10, 10}, {20, 20}, rng);

            //printf("0x%llx\n", *reinterpret_cast<const uint64_t*>(gPtr));
            //printf("0x%llx\n", *(reinterpret_cast<const uint64_t*>(gPtr) + 1));
            //printf("0X%llx\n", *(reinterpret_cast<const uint64_t*>(gPtr) + 2));

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

inline __device__ void AllocateSingleLight(GPULightI*& gPtr,
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

static __global__ void KCAllocateSingleLight(GPULightI* gPtr,
                                             void* const gMemory,
                                             const CPULight cpuLight)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0)
    {
        AllocateSingleLight(gPtr, gMemory, cpuLight);
    }
}

static __global__ void KCAllocateSingleCamera(GPUCameraI*& gPtr,
                                              void* const gMemory,
                                              const CPUCamera cpuCamera)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0)
    {
        AllocateSingleCamera(gPtr, gMemory, cpuCamera);
    }
}


static __global__ void KCAllocateLights(GPULightI** gPtrs,
                                        Byte* const gMemory,
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

static __global__ void KCAllocateCameras(GPUCameraI** gPtrs,
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

inline size_t LightCameraKernels::LightClassesUnionSize()
{
    return GPULightUnionSize;
}

inline size_t LightCameraKernels::CameraClassesUnionSize()
{
    return GPUCameraUnionSize;
}

inline void LightCameraKernels::ConstructLights(// Output
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

inline void LightCameraKernels::ConstructCameras(// Output
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
                            static_cast<uint32_t>(CameraClassesUnionSize()));
}

inline void LightCameraKernels::ConstructSingleLight(// Output
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

inline void LightCameraKernels::ConstructSingleCamera(// Output
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