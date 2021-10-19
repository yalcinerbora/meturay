#pragma once

#include "RayTracer.h"
#include "WorkPool.h"
#include "GPULightI.h"
#include "Tracers.h"

class GPUDirectLightSamplerI;

class PathTracer final : public RayTracer
{
    public:
        static constexpr const char*    TypeName() { return "PathTracer"; }

        static constexpr const char*    MAX_DEPTH_NAME = "MaxDepth";
        static constexpr const char*    NEE_NAME = "NextEventEstimation";
        static constexpr const char*    RR_START_NAME = "RussianRouletteStart";
        static constexpr const char*    DIRECT_LIGHT_MIS_NAME = "DirectLightMIS";
        static constexpr const char*    LIGHT_SAMPLER_TYPE_NAME = "NEESampler";

        struct Options
        {
            int32_t             sampleCount         = 1;
            uint32_t            maximumDepth        = 10;
            bool                nextEventEstimation = true;
            uint32_t            rrStart             = 3;
            bool                directLightMIS      = false;
            LightSamplerType    lightSamplerType    = LightSamplerType::UNIFORM;
        };

    private:
        Options                         options;
        uint32_t                        currentDepth;
        WorkBatchMap                    workMap;
        // Work Pools
        BoundaryWorkPool<bool, bool>    boundaryWorkPool;
        // Single large kernel
        WorkPool<bool, bool>            pathWorkPool;
        // Light Sampler Memory and Pointer
        DeviceMemory                    lightSamplerMemory;
        const GPUDirectLightSamplerI*   dLightSampler;

    protected:
    public:
        // Constructors & Destructor
                                PathTracer(const CudaSystem&,
                                           const GPUSceneI&,
                                           const TracerParameters&);
                                ~PathTracer() = default;

        TracerError             Initialize() override;
        TracerError             SetOptions(const TracerOptionsI&) override;
        void                    AskOptions() override;

        void                    GenerateWork(uint32_t cameraIndex) override;
        void                    GenerateWork(const VisorTransform&, uint32_t cameraIndex) override;
        void                    GenerateWork(const GPUCameraI&) override;
        bool                    Render() override;
};

static_assert(IsTracerClass<PathTracer>::value,
              "PathTracer is not a Tracer Class.");