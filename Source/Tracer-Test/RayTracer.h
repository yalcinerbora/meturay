#pragma once

#include "TracerLib/GPUTracer.h"
#include "TracerLib/TypeTraits.h"
#include "TracerLib/GPUEndpointI.cuh"

#include "RayAuxStruct.h"

// Generic Ray Tracer Class
class RayTracer : public GPUTracer

{
    public:
        // Option Names
        static constexpr const char* SAMPLE_NAME = "Samples";

    private:
        // GPU Image of the camera
        DeviceMemory            cameraMemory;
    protected:
        // Auxiliary Data for Each Ray
        DeviceMemory            auxBuffer0;
        DeviceMemory            auxBuffer1;
        //
        DeviceMemory*           dAuxIn;
        DeviceMemory*           dAuxOut;
        // Camera Realted Ptrs
        GPUCameraI**            dCustomCamera;
        const GPUCameraI**      dSceneCameras;
        Byte*                   dCustomCameraAlloc;
        Byte*                   dSceneCameraAllocs;

        const GPUSceneI&        scene;

        void                    GenerateRays(const GPUCameraI* dCamera, int32_t sampleCount);
        void                    LoadCameraToGPU(const CPUCamera&);
        void                    SwapAuxBuffers();

    public:
        // Constructors & Destructor
                                RayTracer(const CudaSystem&, 
                                          const GPUSceneI&, 
                                          const TracerParameters&);
                                ~RayTracer() = default;

        TracerError             Initialize() override;
};



