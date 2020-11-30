#pragma once

#include "TracerLib/GPUTracer.h"
#include "TracerLib/TypeTraits.h"
#include "TracerLib/GPUEndpointI.cuh"
#include "TracerLib/GPUCameraI.cuh"
#include "TracerLib/CameraFunctions.h"

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

        template <class AuxStruct, AuxInitFunc<AuxStruct> AuxFunc>
        void                    GenerateRays(const GPUCameraI* dCamera, int32_t sampleCount,
                                             const AuxStruct& initialValues);
        template <class AuxStruct, AuxInitFunc<AuxStruct> AuxFunc>
        void                    GenerateRays(const VisorCamera& camera, int32_t sampleCount,
                                             const AuxStruct& initialValues);
        void                    SwapAuxBuffers();

    public:
        // Constructors & Destructor
                                RayTracer(const CudaSystem&, 
                                          const GPUSceneI&, 
                                          const TracerParameters&);
                                ~RayTracer() = default;

        TracerError             Initialize() override;
};