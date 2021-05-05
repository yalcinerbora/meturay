#pragma once

#include "RayTracer.h"
#include "WorkPool.h"
#include "GPULightI.h"

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
        
        enum LightSamplerType
        {
            UNIFORM,

            END
        };

        struct Options
        {
            int32_t             sampleCount         = 1;
            uint32_t            maximumDepth        = 10;
            bool                nextEventEstimation = true;
            uint32_t            rrStart             = 3;
            bool                directLightMIS      = false;
            std::string         lightSamplerType    = "Uniform";
        };

    private:
        Options                         options;
        uint32_t                        currentDepth;
        WorkBatchMap                    workMap;
        // Work Pools
        WorkPool<bool, bool, bool>      boundaryWorkPool;
        WorkPool<>                      pathWorkPool;
        WorkPool<bool>                  neeWorkPool;
        WorkPool<>                      misWorkPool;       
        // Light Sampler Memory and Pointer
        DeviceMemory                    memory;
        const GPUDirectLightSamplerI*   lightSampler;
        // Misc
        static TracerError              LightSamplerNameToEnum(LightSamplerType&,
                                                               const std::string&);
        TracerError                     ConstructLightSampler();

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

        void                    GenerateWork(int cameraId) override;
        void                    GenerateWork(const VisorCamera&) override;
        bool                    Render() override;
};

static_assert(IsTracerClass<PathTracer>::value,
              "PathTracer is not a Tracer Class.");