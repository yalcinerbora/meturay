#pragma once

#include "RayTracer.h"
#include "WorkPool.h"
#include "GPULightI.h"
#include "STree.cuh"
#include "Tracers.h"

class GPUDirectLightSamplerI;
struct PathGuidingNode;

class PPGTracer final : public RayTracer
{
    public:
        static constexpr const char* TypeName() { return "PPGTracer"; }

        static constexpr const char* MAX_DEPTH_NAME             = "MaxDepth";
        static constexpr const char* SAMPLE_NAME                = "Samples";
        static constexpr const char* RR_START_NAME              = "RussianRouletteStart";
        static constexpr const char* LIGHT_SAMPLER_TYPE_NAME    = "NEESampler";

        static constexpr const char* NEE_NAME                   = "NextEventEstimation";
        static constexpr const char* DIRECT_LIGHT_MIS_NAME      = "DirectLightMIS";

        static constexpr const char* RAW_PG_NAME                = "RawPathGuiding";
        static constexpr const char* ALWAYS_SEND_NAME           = "AlwaysSendSamples";
        static constexpr const char* D_TREE_MAX_DEPTH_NAME      = "DTreeMaximumDepth";
        static constexpr const char* D_TREE_FLUX_RATIO_NAME     = "DTreeFluxRatio";
        static constexpr const char* S_TREE_SAMPLE_SPLIT_NAME   = "STreeMaxSamples";
        static constexpr const char* SD_TREE_PATH_NAME          = "SDTreePath";
        static constexpr const char* DUMP_DEBUG_NAME            = "DumpDebugData";

        struct Options
        {
            int32_t             sampleCount         = 1;
            uint32_t            maximumDepth        = 10;

            uint32_t            rrStart             = 3;

            LightSamplerType    lightSamplerType    = LightSamplerType::UNIFORM;

            // Paper Related
            uint32_t            maxDTreeDepth       = 32;
            uint32_t            maxSDTreeSizeMB     = 512;
            uint32_t            sTreeSplitThreshold = 12000;
            float               dTreeSplitThreshold = 0.01f;

            // Initial Tree
            std::string         sdTreePath          = "";

            // Misc
            bool                alwaysSendSamples   = false;
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
        // Global STree
        std::unique_ptr<STree>          sTree;
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
                                PPGTracer(const CudaSystem&,
                                           const GPUSceneI&,
                                           const TracerParameters&);
                                ~PPGTracer() = default;

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

static_assert(IsTracerClass<PPGTracer>::value,
              "PPGTracer is not a Tracer Class.");
