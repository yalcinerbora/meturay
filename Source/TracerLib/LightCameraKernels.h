#pragma once

#include <vector>
#include "RayLib/Types.h"

class CudaSystem;
class GPULightI;
struct CPULight;

class GPUEndpointI;
using GPUCameraI = GPUEndpointI;
struct CPUCamera;

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