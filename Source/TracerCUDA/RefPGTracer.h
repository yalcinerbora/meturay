#pragma once

#include "PathTracer.h"
#include "DirectTracer.h"

#include "WorkPool.h"
#include "GPULightI.h"
#include "STree.cuh"

#include "RayLib/TracerCallbacksI.h"
#include "RayLib/AnalyticData.h"
#include "RayLib/VisorCamera.h"

class GPUCameraSpherical;

class PathTracerMiddleCallback : public TracerCallbacksI
{
    private:
        TracerCallbacksI*       callbacks;

        // Image Related
        Vector2i                resolution;

        // Output Name
        std::vector<float>      lumPixels;
        std::vector<uint32_t>   totalSampleCounts;

    public:
        // Constructors & Destructor
                PathTracerMiddleCallback(const Vector2i& resolution);
                ~PathTracerMiddleCallback() = default;

        // Methods
        void    SaveImage(const std::string& baseName, int pixelId);
        void    SetCallbacks(TracerCallbacksI*);

        // Interface
        void    SendCrashSignal() override {};
        void    SendLog(const std::string) override;
        void    SendError(TracerError) override;
        void    SendAnalyticData(AnalyticData) override {};
        void    SendImageSectionReset(Vector2i start = Zero2i,
                                      Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void    SendImage(const std::vector<Byte> data,
                          PixelFormat, size_t offset,
                          Vector2i start = Zero2i,
                          Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void    SendCurrentOptions(TracerOptions) override;
        void    SendCurrentParameters(TracerParameters) override {};
        void    SendCurrentCamera(VisorCamera) override {};
        void    SendCurrentSceneCameraCount(uint32_t) override {};
};

class DirectTracerMiddleCallback : public TracerCallbacksI
{
    private:
        std::vector<Vector3f>   pixelLocations;
        Vector2i                portionStart;
        Vector2i                portionEnd;
        Vector2i                resolution;

    public:
        // Constructors & Destructor
                DirectTracerMiddleCallback() = default;

        //  Interface
        void    SendCrashSignal() override {};
        void    SendLog(const std::string) override;
        void    SendError(TracerError) override;
        void    SendAnalyticData(AnalyticData) override {};
        void    SendImageSectionReset(Vector2i start = Zero2i,
                                      Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void    SendImage(const std::vector<Byte> data,
                          PixelFormat, size_t offset,
                          Vector2i start = Zero2i,
                          Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void    SendCurrentOptions(TracerOptions) override;
        void    SendCurrentParameters(TracerParameters) override {};
        void    SendCurrentCamera(VisorCamera) override {};
        void    SendCurrentSceneCameraCount(uint32_t) override {};

        // Setters
        void                SetSection(const Vector2i&, const Vector2i&);
        void                SetResolution(const Vector2i&);

        const Vector3f&     Pixel(uint32_t pixelIndex) const;
        const Vector2i&     Start() const;
        const Vector2i&     End() const;
        const Vector2i&     Resolution() const;
};

class RefPGTracer : public GPUTracerI
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

            std::string         lightSamplerType    = "Uniform";
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
        // Tracers
        PathTracer                      pathTracer;
        DirectTracer                    directTracer;
        // Callbacks
        TracerCallbacksI*               callbacks;
        bool                            crashed;
        // Params
        TracerParameters                params;
        //
        const CudaSystem&               cudaSystem;
        //
        const GPUSceneI&                scene;
        // Callbacks for each Tracer
        DirectTracerMiddleCallback      dtCallbacks;
        PathTracerMiddleCallback        ptCallbacks;
        // Spherical Camera (for PT Rendering)
        DeviceMemory                    memory;
        GPUCameraSpherical*             dSphericalCamera;
        bool                            doInitCameraCreation;

        std::unique_ptr<TracerOptionsI> GenerateDirectTracerOptions();

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
        void                    GenerateWork(const GPUCameraI&) override;
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
    ptCallbacks.SetCallbacks(callbacks);
}

inline const Vector3f& DirectTracerMiddleCallback::Pixel(uint32_t pixelIndex) const
{
    return pixelLocations[pixelIndex];
}

inline void DirectTracerMiddleCallback::SetSection(const Vector2i& start,
                                                   const Vector2i& end)
{
    portionStart = start;
    portionEnd = end;
    portionEnd = Vector2i::Min(portionEnd, resolution);

    // Allocate an image section
    Vector2i size = portionEnd - portionStart;
    pixelLocations.resize(size[0] * size[1]);
}

inline void DirectTracerMiddleCallback::SetResolution(const Vector2i& v)
{
    resolution = v;
}

inline const Vector2i& DirectTracerMiddleCallback::Start() const
{
    return portionStart;
}

inline const Vector2i& DirectTracerMiddleCallback::End() const
{
    return portionEnd;
}

inline const Vector2i& DirectTracerMiddleCallback::Resolution() const
{
    return resolution;
}

static_assert(IsTracerClass<RefPGTracer>::value,
              "RefPGTracer is not a Tracer Class.");
