#pragma once

#include "RayTracer.h"
#include "WorkPool.h"
#include "GPULightI.h"
#include "Tracers.h"
#include "AnisoSVO.cuh"
#include "WFPGCommon.h"

#include "RNGSobol.cuh"
#include "RNGIndependent.cuh"

class GPUDirectLightSamplerI;

class WFPGTracer final : public RayTracer
{
    public:
        static constexpr const char*    TypeName() { return "WFPGTracer"; }

        static constexpr const char*    MAX_DEPTH_NAME          = "MaxDepth";
        static constexpr const char*    NEE_NAME                = "NextEventEstimation";
        static constexpr const char*    RR_START_NAME           = "RussianRouletteStart";
        static constexpr const char*    DIRECT_LIGHT_MIS_NAME   = "DirectLightMIS";
        static constexpr const char*    LIGHT_SAMPLER_TYPE_NAME = "NEESampler";

        static constexpr const char*    OCTREE_LEVEL_NAME       = "OctreeLevel";
        static constexpr const char*    RAY_BIN_MIN_LEVEL_NAME  = "MinRayBinLevel";
        static constexpr const char*    BIN_RAY_COUNT_NAME      = "BinRayCount";

        static constexpr const char*    RENDER_MODE_NAME        = "RenderMode";
        static constexpr const char*    SVO_DEBUG_ITER_NAME     = "SVODebugRenderIter";
        static constexpr const char*    DUMP_DEBUG_NAME         = "DumpDebugData";
        static constexpr const char*    DUMP_INTERVAL_NAME      = "DataDumpIntervalExp";

        struct Options
        {
            int32_t             sampleCount         = 1;
            uint32_t            maximumDepth        = 10;
            bool                nextEventEstimation = true;
            uint32_t            rrStart             = 3;
            bool                directLightMIS      = false;
            LightSamplerType    lightSamplerType    = LightSamplerType::UNIFORM;

            // Method Related
            uint32_t            octreeLevel         = 10;   // Octree Level (10 = 1024x1024x1024)
            uint32_t            minRayBinLevel      = 5;    // When rays are binned they cannot group
                                                            // even if they did not satisfy the ray bin count
            uint32_t            binRayCount         = 512;  // Amount of rays on each bin

            // Photon
            uint32_t            photonPerPass       = 1'000'000;

            // Misc
            WFPGRenderMode      renderMode          = WFPGRenderMode::NORMAL;
            uint32_t            svoRadRenderIter    = 2;
            bool                dumpDebugData       = false;
            uint32_t            svoDumpInterval     = 2;
        };

    private:
        Options                         options;
        uint32_t                        currentDepth;
        WorkBatchMap                    workMap;
        WorkBatchMap                    photonWorkMap;
         // Work Pools
        BoundaryWorkPool<bool, bool>    boundaryWorkPool;
        WorkPool<bool, bool>            pathWorkPool;
        // Debug Works
        BoundaryWorkPool<>              debugBoundaryWorkPool;
        WorkPool<>                      debugPathWorkPool;
        // Photon Works
        WorkPool<>                      photonWorkPool;

        // Light Sampler Memory and Pointer
        DeviceMemory                    lightSamplerMemory;
        const GPUDirectLightSamplerI*   dLightSampler;

        uint32_t                        iterationCount;
        uint32_t                        treeDumpCount;
        // SVO
        AnisoSVOctreeCPU                svo;
        // RNG and Guide Sample Related
        //RNGScrSobolCPU                  pgSampleRNG;
        RNGIndependentCPU               pgSampleRNG;
        uint32_t                        pgKernelBlockCount;
        // Path Memory
        DeviceMemory                    pathMemory;
        PathGuidingNode*                dPathNodes;
        // Misc
        void                            ResizeAndInitPathMemory();
        uint32_t                        TotalPathNodeCount() const;
        uint32_t                        MaximumPathNodePerPath() const;

        void                            GenerateGuidedDirections();
        void                            TraceAndStorePhotons();

        // TODO: Change
        // Temporarily store the current camera
        // For SVO_RADIANCE MOde
        enum CameraType
        {
            SCENE_CAMERA,
            CUSTOM_CAMERA,
            TRANSFORMED_SCENE_CAMERA
        };
        struct
        {
            CameraType type;
            union
            {
                const GPUCameraI* dCustomCamera;
                struct
                {
                    uint32_t cameraIndex;
                    VisorTransform transform;
                } transformedSceneCam;
                uint32_t nonTransformedCamIndex;
            };
        } currentCamera;

    protected:
    public:
        // Constructors & Destructor
                                WFPGTracer(const CudaSystem&,
                                           const GPUSceneI&,
                                           const TracerParameters&);
                                ~WFPGTracer() = default;

        TracerError             Initialize() override;
        TracerError             SetOptions(const OptionsI&) override;
        void                    AskOptions() override;

        void                    GenerateWork(uint32_t cameraIndex) override;
        void                    GenerateWork(const VisorTransform&, uint32_t cameraIndex) override;
        void                    GenerateWork(const GPUCameraI&) override;
        bool                    Render() override;
        void                    Finalize() override;

        size_t                  TotalGPUMemoryUsed() const override;
};

static_assert(IsTracerClass<WFPGTracer>::value,
              "WFPGTracer is not a Tracer Class.");