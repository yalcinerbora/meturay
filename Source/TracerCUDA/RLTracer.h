#pragma once

#include "RayTracer.h"
#include "WorkPool.h"
#include "GPULightI.h"
#include "SceneSurfaceTree.cuh"
#include "Dense2DArray.cuh"
#include "Tracers.h"

class GPUDirectLightSamplerI;
struct PathGuidingNode;

class RLTracer final : public RayTracer
{
    public:
        static constexpr const char* TypeName() { return "RLTracer"; }

        static constexpr const char* MAX_DEPTH_NAME             = "MaxDepth";
        static constexpr const char* SAMPLE_NAME                = "Samples";
        static constexpr const char* RR_START_NAME              = "RussianRouletteStart";
        static constexpr const char* LIGHT_SAMPLER_TYPE_NAME    = "NEESampler";

        static constexpr const char* NEE_NAME                   = "NextEventEstimation";
        static constexpr const char* DIRECT_LIGHT_MIS_NAME      = "DirectLightMIS";

        static constexpr const char* RAW_PG_NAME                = "RawPathGuiding";
        static constexpr const char* DIRECTONAL_RES_NAME        = "DirectionalResolution";
        static constexpr const char* NORM_THRESHOLD_NAME        = "NormalThreshold";
        static constexpr const char* SPATIAL_SAMPLE_NAME        = "SpatialSampleCount";
        static constexpr const char* ALPHA_NAME                 = "Alpha";

        static constexpr const char* DUMP_DEBUG_NAME            = "DumpDebugData";

        struct Options
        {
            int32_t             sampleCount         = 1;
            uint32_t            maximumDepth        = 10;

            uint32_t            rrStart             = 3;

            LightSamplerType    lightSamplerType    = LightSamplerType::UNIFORM;

            // Paper Related
            Vector2i            directionalRes      = Vector2i(16, 16);
            uint32_t            spatialSamples      = 2048;
            float               alpha               = 0.5f;
            float               normalThreshold     = 45.0f; // degrees

            // Misc
            bool                rawPathGuiding      = true;

            bool                nextEventEstimation = true;
            bool                directLightMIS      = false;

            bool                dumpDebugData       = false;
        };

    private:
        Options                         options;
        uint32_t                        currentDepth;
        WorkBatchMap                    workMap;
        // Work Pools
        BoundaryWorkPool<bool, bool>    boundaryWorkPool;
        WorkPool<bool, bool>            pathWorkPool;
        // Light Sampler Memory and Pointer
        DeviceMemory                    lightSamplerMemory;
        const GPUDirectLightSamplerI*   dLightSampler;
        // Path Memory
        DeviceMemory                    pathMemory;
        PathGuidingNode*                dPathNodes;
        // Global Data Structure
        DeviceMemory                    memory;
        SceneSurfaceTree                surfaceTree;
        Dense2DArrayCPU                 denseArray;
        // Internal State
        uint32_t                        currentTreeIteration;
        uint32_t                        nextTreeSwap;
        // Misc
        void                            ResizeAndInitPathMemory();
        uint32_t                        TotalPathNodeCount() const;
        uint32_t                        MaximumPathNodePerPath() const;

    protected:
    public:
        // Constructors & Destructor
                                RLTracer(const CudaSystem&,
                                           const GPUSceneI&,
                                           const TracerParameters&);
                                ~RLTracer() = default;

        TracerError             Initialize() override;
        TracerError             SetOptions(const TracerOptionsI&) override;
        void                    AskOptions() override;

        void                    GenerateWork(uint32_t cameraIndex) override;
        void                    GenerateWork(const VisorTransform&, uint32_t cameraIndex) override;
        void                    GenerateWork(const GPUCameraI&) override;
        bool                    Render() override;
        void                    Finalize() override;

        size_t                  TotalGPUMemoryUsed() const override;
};

static_assert(IsTracerClass<RLTracer>::value,
              "RLTracer is not a Tracer Class.");
