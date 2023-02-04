#pragma once

#include "RayTracer.h"
#include "WorkPool.h"
#include "GPULightI.h"
#include "Tracers.h"
#include "AnisoSVO.cuh"
#include "WFPGCommon.h"

#include "RNGSobol.cuh"
#include "RNGIndependent.cuh"
#include "BlockTextureFilter.cuh"
#include "GPUMetaSurfaceGenerator.h"

class GPUDirectLightSamplerI;
class SVOOptixConeCaster;
class SVOOptixRadianceBuffer;

class WFPGTracer final : public RayTracer
{
    public:
    static constexpr const char*    TypeName() { return "WFPGTracer"; }

    // PT Related
    static constexpr const char*    MAX_DEPTH_NAME              = "MaxDepth";
    static constexpr const char*    NEE_NAME                    = "NextEventEstimation";
    static constexpr const char*    RR_START_NAME               = "RussianRouletteStart";
    static constexpr const char*    DIRECT_LIGHT_MIS_NAME       = "DirectLightMIS";
    static constexpr const char*    LIGHT_SAMPLER_TYPE_NAME     = "NEESampler";

    //
    static constexpr const char*    OCTREE_LEVEL_NAME           = "OctreeLevel";
    static constexpr const char*    RAY_BIN_MIN_LEVEL_NAME      = "MinRayBinLevel";
    static constexpr const char*    BIN_RAY_COUNT_NAME          = "BinRayCount";

    static constexpr const char*    RENDER_MODE_NAME            = "RenderMode";
    static constexpr const char*    SVO_DEBUG_ITER_NAME         = "SVODebugRenderIter";
    static constexpr const char*    RENDER_LEVEL_NAME           = "SVORenderLevel";
    static constexpr const char*    SVO_INIT_PATH_NAME          = "InitialSVO";
    static constexpr const char*    SKIP_PG_NAME                = "SkipPG";
    static constexpr const char*    PURE_PG_NAME                = "PurePG";
    static constexpr const char*    MIS_RATIO_NAME              = "BXDF-GuideMISRatio";
    static constexpr const char*    PRODUCT_PG_NAME             = "DoProductPathGuiding";
    static constexpr const char*    OPTIX_TRACE_NAME            = "OptiXTraceSVO";
    static constexpr const char*    OPTIX_BUFFER_NAME           = "OptixTraceBufferSize";

    static constexpr const char*    R_FIELD_GAUSS_ALPHA_NAME    = "RFieldFilterAlpha";

    static constexpr const char*    PG_DUMP_DEBUG_NAME          = "PGDumpDataStruct";
    static constexpr const char*    PG_DUMP_INTERVAL_NAME       = "PGDataStructDumpIntervalExp";
    static constexpr const char*    PG_DUMP_PATH_NAME           = "PGDataStructDumpName";

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

        // Misc
        WFPGRenderMode      renderMode          = WFPGRenderMode::RENDER;
        uint32_t            svoRadRenderIter    = 2;
        uint32_t            svoRenderLevel      = 0;
        std::string         svoInitPath         = "";
        float               rFieldGaussAlpha    = 1.0f; // Filter of the radiance field
        bool                skipPG              = false;
        bool                purePG              = false;
        bool                productPG           = true;
        float               misRatio            = 0.5f;
        bool                optiXTrace          = false;
        uint32_t            optiXBufferSize     = 16;
        //
        bool                pgDumpDebugData     = false;
        uint32_t            pgDumpInterval      = 2;
        std::string         pgDumpDebugName     = "wfpg_svo";
    };

    private:
        Options                         options;
        uint32_t                        currentDepth;
        WorkBatchMap                    workMap;
         // Work Pools
        BoundaryWorkPool<bool, bool>    boundaryWorkPool;
        WorkPool<bool, bool>            pathWorkPool;
        // Debug Works
        BoundaryWorkPool<>              debugBoundaryWorkPool;
        WorkPool<>                      debugPathWorkPool;
        // Device filter for radiance field
        GaussFilter                     rFieldGaussFilter;

        // OptiX cone caster (allocated if requested)
        std::unique_ptr<SVOOptixConeCaster>     coneCasterOptiX;
        std::unique_ptr<SVOOptixRadianceBuffer> radianceBufferOptiX;
        std::unique_ptr<DeviceMemory>           binInfoBufferOptiX;

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
        WFPGPathNode*                   dPathNodes;
        // SVO Partition/Reduction Memory
        DeviceMemory                    partitionMemory;
        // Product Path Guiding Related
        GPUMetaSurfaceHandler           metaSurfHandler;
        // Misc
        void                            ResizeAndInitPathMemory();
        uint32_t                        TotalPathNodeCount() const;
        uint32_t                        MaximumPathNodePerPath() const;

        void                            AccumulateRayHitsToSVO();
        void                            GenerateGuidedDirections();
        void                            LaunchDebugConeTraceKernel();

        // TODO: Change bad design
        // Temporarily store the current camera
        // For SVO_RADIANCE Mode
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