#pragma once

#include "RayTracer.h"

#include "WorkPool.h"
#include "GPULightI.h"
#include "STree.cuh"
#include "Tracers.h"

#include "RayLib/TracerCallbacksI.h"
#include "RayLib/AnalyticData.h"

struct ImageIOError;
class GPUCameraPixel;

class RefPGTracer : public RayTracer
{
    public:
        static constexpr const char* TypeName() { return "RefPGTracer"; }

        static constexpr const char* MAX_DEPTH_NAME             = "MaxDepth";
        static constexpr const char* SAMPLE_NAME                = "Samples";
        static constexpr const char* RR_START_NAME              = "RussianRouletteStart";
        static constexpr const char* LIGHT_SAMPLER_TYPE_NAME    = "NEESampler";
        static constexpr const char* MAX_SAMPLE_NAME            = "TotalSamplePerPixel";
        static constexpr const char* RESOLUTION_NAME            = "Resolution";
        static constexpr const char* IMAGE_NAME                 = "ImageName";

        static constexpr const char* NEE_NAME                   = "NextEventEstimation";
        static constexpr const char* DIRECT_LIGHT_MIS_NAME      = "DirectLightMIS";


        struct Options
        {
            uint32_t            totalSamplePerPixel = 65536;
            int32_t             samplePerIteration  = 1;
            uint32_t            maximumDepth        = 10;

            uint32_t            rrStart             = 3;

            //
            Vector2i            resolution          = Vector2i(1024);
            std::string         refPGOutputName     = "refPGOut";

            LightSamplerType    lightSamplerType    = LightSamplerType::UNIFORM;
            // Misc
            bool                nextEventEstimation = true;
            bool                directLightMIS      = false;
        };

    private:
        Options                         options;
        // Internal State
        uint32_t                        currentPixel;
        uint32_t                        currentSampleCount;
        int                             currentCamera;
        uint32_t                        currentDepth;
        // Works
        WorkBatchMap                    workMap;
        // Work Kernel Loaders
        BoundaryWorkPool<bool, bool>    boundaryWorkPool;
        WorkPool<bool, bool>            pathWorkPool;
        // Light Sampler Memory and Pointer
        DeviceMemory                    lsMemory;
        const GPUDirectLightSamplerI*   dLightSampler;
        // Pixel Camera
        DeviceMemory                    camMemory;
        GPUCameraPixel*                 dPixelCamera;
        bool                            doInitCameraCreation;
        // Image Related
        Vector2i                        resolution;     // Image Resolution
        Vector2i                        iPortionStart;  // Our image portion
        Vector2i                        iPortionEnd;
        PixelFormat                     iPixelFormat;    // Requested Pixel Format
        //
        ImageMemory                     accumulationBuffer;


        // Misc Methods
        void                            SendPixel() const;
        Vector2i                        GlobalPixel2D() const;
        void                            ResetIterationVariables();
        ImageIOError                    SaveAndResetAccumImage(const Vector2i& pixelId);

    protected:
    public:
        // Constructors & Destructor
                                RefPGTracer(const CudaSystem&,
                                            const GPUSceneI&,
                                            const TracerParameters&);
                                ~RefPGTracer() = default;

        TracerError             Initialize() override;
        TracerError             SetOptions(const TracerOptionsI&) override;
        void                    AskOptions() override;

        void                    GenerateWork(uint32_t cameraIndex) override;
        void                    GenerateWork(const VisorTransform&, uint32_t cameraIndex) override;
        void                    GenerateWork(const GPUCameraI&) override;
        bool                    Render() override;
        void                    Finalize() override;

        void                    AskParameters() override;
        // Image Related Commands
        void                    SetImagePixelFormat(PixelFormat) override;
        void                    ReportionImage(Vector2i start = Zero2i,
                                               Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void                    ResizeImage(Vector2i resolution) override;
        void                    ResetImage() override;
};

static_assert(IsTracerClass<RefPGTracer>::value,
              "RefPGTracer is not a Tracer Class.");
