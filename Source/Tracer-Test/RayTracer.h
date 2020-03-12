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
    protected:
        DeviceMemory            tempCameraBuffer;
        DeviceMemory            auxBuffer0;
        DeviceMemory            auxBuffer1;

        GPUCameraI**            dCameraPtr;
        DeviceMemory*           dAuxIn;
        DeviceMemory*           dAuxOut;        

        GPUSceneI&              scene;

        void                    GenerateRays(const GPUCameraI& dCamera, int32_t sampleCount);
        void                    LoadCameraToGPU(const CPUCamera&);
        void                    SwapAuxBuffers();

    public:
        // Constructors & Destructor
                                RayTracer(CudaSystem&, GPUSceneI&, 
                                            const TracerParameters&);
                                ~RayTracer() = default;

        TracerError             Initialize() override;
};



