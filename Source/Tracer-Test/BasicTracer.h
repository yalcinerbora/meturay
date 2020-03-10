#pragma once

#include "TracerLib/GPUTracer.h"
#include "TracerLib/TypeTraits.h"
#include "TracerLib/GPUEndpointI.cuh"

#include "RayAuxStruct.h"

class BasicTracer final : public GPUTracer

{
    public:
        static constexpr const char*    TypeName() { return "TestBasic"; }

        // Option Names
        static constexpr const char*    SAMPLE_NAME = "Samples";
        static constexpr const char*    MAX_DEPTH_NAME = "MaxDepth";

        struct Options                 
        {
            int32_t     sampleCount;    // Per-axis sample per pixel
            uint32_t    maximumDepth;
        };

    private:
        DeviceMemory            auxBuffer0;
        DeviceMemory            auxBuffer1;

        DeviceMemory*           auxIn;
        DeviceMemory*           auxOut;        

        GPUSceneI&              scene;

        Options                 options;

        void                    GenerateRays(const GPUCameraI& dCamera);
        void                    SwapAuxBuffers();

    protected:
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