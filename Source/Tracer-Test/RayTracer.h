#pragma once

#include "TracerLib/GPUTracer.h"
#include "TracerLib/TypeTraits.h"
#include "TracerLib/GPUEndpointI.cuh"
#include "TracerLib/GPUCameraI.cuh"

#include "RayAuxStruct.h"

class GPUCameraVisor;

// Generic Ray Tracer Class
class RayTracer : public GPUTracer

{
    public:
        // Option Names
        static constexpr const char* SAMPLE_NAME = "Samples";

    private:
    protected:
        // Auxiliary Data for Each Ray
        DeviceMemory            auxBuffer0;
        DeviceMemory            auxBuffer1;
        //
        DeviceMemory*           dAuxIn;
        DeviceMemory*           dAuxOut;

        const GPUSceneI&        scene;

        virtual void            GenerateRays(const GPUCameraI* dCamera, int32_t sampleCount);
        virtual void            GenerateRays(const VisorCamera& camera, int32_t sampleCount);
        void                    SwapAuxBuffers();

    public:
        // Constructors & Destructor
                                RayTracer(const CudaSystem&, 
                                          const GPUSceneI&, 
                                          const TracerParameters&);
                                ~RayTracer() = default;

        TracerError             Initialize() override;
};



