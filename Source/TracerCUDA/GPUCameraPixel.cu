#include "GPUCameraPixel.cuh"
#include "CudaSystem.hpp"
#include "RayLib/MemoryAlignment.h"

__global__
void KCConstructSingleGPUCameraPixel(GPUCameraPixel* gCameraLocations,
                                     bool deletePrevious,
                                     //
                                     const GPUCameraI& baseCam,
                                     int32_t pixelIndex,
                                     Vector2i resolution)
{
    if(threadIdx.x != 0) return;

    if(deletePrevious) gCameraLocations->~GPUCameraPixel();
    new (gCameraLocations) GPUCameraPixel(baseCam,
                                          pixelIndex,
                                          resolution);
}