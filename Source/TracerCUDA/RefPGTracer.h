#pragma once

#include "PathTracer.h"
#include "DirectTracer.h"

#include "WorkPool.h"
#include "GPULightI.h"
#include "STree.cuh"

class GPUCameraSpherical;

class RefPGTracer : public GPUTracerI
{
    public:
        static constexpr const char* TypeName() { return "RefPGTracer"; }

        static constexpr const char* MAX_DEPTH_NAME             = "MaxDepth";
        static constexpr const char* SAMPLE_NAME                = "Samples";        
        static constexpr const char* RR_START_NAME              = "RussianRouletteStart";        
        static constexpr const char* LIGHT_SAMPLER_TYPE_NAME    = "NEESampler";
        static constexpr const char* MAX_SAMPLE_NAME            = "MaxSamples";
        
        static constexpr const char* NEE_NAME                   = "NextEventEstimation";
        static constexpr const char* DIRECT_LIGHT_MIS_NAME      = "DirectLightMIS";

        struct Options
        {
            uint32_t            maxSampleCount      = 65536;
            int32_t             sampleCount         = 1;
            uint32_t            maximumDepth        = 10;

            uint32_t            rrStart             = 3;

            //
            Vector2i            resolution          = Vector2i(1024);

            std::string         lightSamplerType    = "Uniform";
            // Misc
            bool                nextEventEstimation = true;            
            bool                directLightMIS      = false;            
        };

    private:
        Options                         options;              
        // Internal State
        uint32_t                        currentPixel;
        uint32_t                        currentDepth;
        uint32_t                        currentSample;
        // Tracers
        PathTracer                      pathTracer;
        DirectTracer                    directTracer;
        // Callbacks
        TracerCallbacksI*               callbacks;
        bool                            crashed;
        // Params
        TracerParameters                params;
        Vector2i                        portionStart; 
        Vector2i                        portionEnd;
        //
        const CudaSystem&               cudaSystem;
        // List of Pixel Locations
        std::vector<Vector3f>           pixelLocations;
        // Spherical Camera (for PT Rendering=
        DeviceMemory                    memory;
        GPUCameraSpherical*             dSphericalCamera;

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

        void                    GenerateWork(int cameraId) override;
        void                    GenerateWork(const VisorCamera&) override;
        bool                    Render() override;
        void                    Finalize() override;

        // Response form Tracer
        void                    AttachTracerCallbacks(TracerCallbacksI&) override;
        // Commands to Tracer
        void                    SetParameters(const TracerParameters&) override;
        void                    AskParameters() override;
        // Image Related Commands
        void                    SetImagePixelFormat(PixelFormat) override;
        void                    ReportionImage(Vector2i start = Zero2i,
                                               Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void                    ResizeImage(Vector2i resolution) override;
        void                    ResetImage() override;
};

inline void RefPGTracer::AttachTracerCallbacks(TracerCallbacksI& tc)
{
    callbacks = &tc;
}

static_assert(IsTracerClass<RefPGTracer>::value,
              "RefPGTracer is not a Tracer Class.");
