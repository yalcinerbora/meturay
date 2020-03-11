#pragma once

#include "TracerLib/GPUTracer.h"
#include "TracerLib/TypeTraits.h"
#include "TracerLib/GPUEndpointI.cuh"

#include "RayAuxStruct.h"

class BasicTracer : public GPUTracer

{
    public:
        static constexpr const char*    TypeName() { return "TestBasic"; }

        // Option Names
        static constexpr const char*    SAMPLE_NAME = "Samples";
        
        struct Options                 
        {
            int32_t     sampleCount = 1;    // Per-axis sample per pixel
        };

    private:
        Options                 options;

    protected:
        DeviceMemory            tempCameraBuffer;
        DeviceMemory            auxBuffer0;
        DeviceMemory            auxBuffer1;

        GPUCameraI**            dCameraPtr;
        DeviceMemory*           dAuxIn;
        DeviceMemory*           dAuxOut;        

        GPUSceneI&              scene;

        WorkBatchMap            workMap;

        void                    GenerateRays(const GPUCameraI& dCamera);
        void                    SwapAuxBuffers();

    public:
        // Constructors & Destructor
                                BasicTracer(CudaSystem&, GPUSceneI&, 
                                            const TracerParameters&);
                                ~BasicTracer() = default;

        TracerError             Initialize() override;
        TracerError             SetOptions(const TracerOptionsI&) override;
        void                    AskOptions() override;

        //
        void                    GenerateWork(int cameraId) override;
        void                    GenerateWork(const CPUCamera&) override;
        bool                    Render() override;
};

static_assert(IsTracerClass<BasicTracer>::value,
              "TracerBasic is not a Tracer Class.");