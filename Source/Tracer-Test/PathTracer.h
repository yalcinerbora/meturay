#pragma once

#include "RayTracer.h"
#include "MetaWorkPool.h"

class PathTracer final : public RayTracer
{
    public:
        static constexpr const char*    TypeName() { return "TestPath"; }

        static constexpr const char*    MAX_DEPTH_NAME = "MaxDepth";
        static constexpr const char*    NEE_NAME = "NextEventEstimation";

        struct Options
        {
            int32_t                     sampleCount = 1;
            uint32_t                    maximumDepth = 10;
            bool                        nextEventEstimation = true;
        };

    private:
        Options                 options;
        uint32_t                currentDepth;
        WorkBatchMap            workMap;
        MetaWorkPool            workPool;
        
        // Light Next Event Estimation
        uint32_t                lightCount;
        DeviceMemory            lightMemory;
        const GPULightI**       dLights;
        Byte*                   dLightAlloc;

    protected:

        void                    GenerateRays(const GPUCameraI* dCamera, int32_t sampleCount) override;

    public:
        // Constructors & Destructor
                                PathTracer(const CudaSystem&,
                                           const GPUSceneI&,
                                           const TracerParameters&);
                                ~PathTracer() = default;

        TracerError             Initialize() override;
        TracerError             SetOptions(const TracerOptionsI&) override;
        void                    AskOptions() override;

        void                    GenerateWork(int cameraId) override;
        void                    GenerateWork(const CPUCamera&) override;
        bool                    Render() override;
};

static_assert(IsTracerClass<PathTracer>::value,
              "TracerBasic is not a Tracer Class.");