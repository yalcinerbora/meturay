#pragma once

#include "RayTracer.h"
#include "WorkPool.h"
#include "GPULightI.h"
#include "STree.cuh"

class GPUDirectLightSamplerI;
struct PathGuidingNode;

class PPGTracer final : public RayTracer
{
    public:
        static constexpr const char*    TypeName() { return "PPGTracer"; }

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

            uint32_t            rrStart             = 3;

            std::string         lightSamplerType    = "Uniform";

            // Paper Related
            uint32_t            maxDTreeDepth       = 32;
            uint32_t            maxSDTreeSizeMB     = 512;
            uint32_t            sTreeSplitThreshold = 12000;
            float               dTreeSplitThreshold = 0.01f;

            // Misc
            bool                alwaysSendSamples   = false;


            bool                nextEventEstimation = true;            
            bool                directLightMIS      = false;
            
        };

        static constexpr bool   USE_SINGLE_PATH_KERNEL = true;

    private:
        Options                         options;
        uint32_t                        currentDepth;
        WorkBatchMap                    workMap;
        // Work Pools
        WorkPool<bool, bool, bool>      boundaryWorkPool;
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

        static TracerError              LightSamplerNameToEnum(LightSamplerType&,
                                                               const std::string&);
        TracerError                     ConstructLightSampler();
        void                            ResizeAndInitPathMemory(size_t pixelCount,
                                                                size_t samplePerPixel);
        uint32_t                        TotalPathNodeCount() const;       

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

        void                    GenerateWork(int cameraId) override;
        void                    GenerateWork(const VisorCamera&) override;
        bool                    Render() override;
        void                    Finalize() override;
};

static_assert(IsTracerClass<PPGTracer>::value,
              "PPGTracer is not a Tracer Class.");