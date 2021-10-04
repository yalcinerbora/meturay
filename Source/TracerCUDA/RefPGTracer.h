#pragma once

#include "RayTracer.h"

#include "WorkPool.h"
#include "GPULightI.h"
#include "STree.cuh"
#include "Tracers.h"

#include "RayLib/TracerCallbacksI.h"
#include "RayLib/AnalyticData.h"
#include "RayLib/VisorCamera.h"

struct ImageIOError;
class GPUCameraPixel;

//class PathTracerMiddleCallback : public TracerCallbacksI
//{
//    private:
//        TracerCallbacksI*       callbacks;
//
//        // Image Related
//        Vector2i                resolution;
//
//        // Output Name
//        std::vector<float>      lumPixels;
//        std::vector<uint32_t>   totalSampleCounts;
//
//    public:
//        // Constructors & Destructor
//                PathTracerMiddleCallback(const Vector2i& resolution);
//                ~PathTracerMiddleCallback() = default;
//
//        // Methods
//        void    SaveImage(const std::string& baseName, Vector2i pixelId);
//        void    SetCallbacks(TracerCallbacksI*);
//
//        // Interface
//        void    SendCrashSignal() override {};
//        void    SendLog(const std::string) override;
//        void    SendError(TracerError) override;
//        void    SendAnalyticData(AnalyticData) override {};
//        void    SendImageSectionReset(Vector2i start = Zero2i,
//                                      Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
//        void    SendImage(const std::vector<Byte> data,
//                          PixelFormat, size_t offset,
//                          Vector2i start = Zero2i,
//                          Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
//        void    SendCurrentOptions(TracerOptions) override;
//        void    SendCurrentParameters(TracerParameters) override {};
//        void    SendCurrentCamera(VisorCamera) override {};
//        void    SendCurrentSceneCameraCount(uint32_t) override {};
//};
//
//class DirectTracerMiddleCallback : public TracerCallbacksI
//{
//    private:
//        std::vector<Vector4f>   pixelLocations;
//        Vector2i                portionStart;
//        Vector2i                portionEnd;
//        Vector2i                resolution;
//
//    public:
//        // Constructors & Destructor
//                    DirectTracerMiddleCallback() = default;
//
//        // Methods
//        Vector2i    PixelGlobalId(int linearLocalPixelId);
//        //  Interface
//        void        SendCrashSignal() override {};
//        void        SendLog(const std::string) override;
//        void        SendError(TracerError) override;
//        void        SendAnalyticData(AnalyticData) override {};
//        void        SendImageSectionReset(Vector2i start = Zero2i,
//                                          Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
//        void        SendImage(const std::vector<Byte> data,
//                              PixelFormat, size_t offset,
//                              Vector2i start = Zero2i,
//                              Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
//        void        SendCurrentOptions(TracerOptions) override;
//        void        SendCurrentParameters(TracerParameters) override {};
//        void        SendCurrentCamera(VisorCamera) override {};
//        void        SendCurrentSceneCameraCount(uint32_t) override {};
//
//        // Setters
//        void                SetSection(const Vector2i&, const Vector2i&);
//        void                SetResolution(const Vector2i&);
//
//        const Vector4f&     Pixel(uint32_t pixelIndex) const;
//        const Vector2i&     Start() const;
//        const Vector2i&     End() const;
//        const Vector2i&     Resolution() const;
//};

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
        WorkPool<bool, bool, bool>      boundaryWorkPool;
        WorkPool<bool, bool>            pathWorkPool;        
        // Light Sampler Memory and Pointer
        DeviceMemory                    lsMemory;
        const GPUDirectLightSamplerI*   dLightSampler;
        // Pixel Camera
        DeviceMemory                    camMemory;
        GPUCameraPixel*                 dPixelCamera;
        bool                            doInitCameraCreation;
        // Image Related
        Vector2i                        resolution;
        Vector2i                        iPortionStart;
        Vector2i                        iPortionEnd;
        //
        ImageMemory                     accumulationBuffer;


        // Misc Methods
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

        void                    GenerateWork(int cameraId) override;
        void                    GenerateWork(const VisorCamera&) override;
        void                    GenerateWork(const GPUCameraI&) override;
        bool                    Render() override;
        void                    Finalize() override;

        void                    AskParameters() override;
        // Image Related Commands
        void                    ReportionImage(Vector2i start = Zero2i,
                                               Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void                    ResizeImage(Vector2i resolution) override;
};

//inline Vector2i DirectTracerMiddleCallback::PixelGlobalId(int linearLocalPixelId)
//{
//    Vector2i segmentSize = portionEnd - portionStart;
//    Vector2i localPixelId2D(linearLocalPixelId % segmentSize[0],
//                            linearLocalPixelId / segmentSize[0]);
//    return portionStart + localPixelId2D;
//}
//
//inline const Vector4f& DirectTracerMiddleCallback::Pixel(uint32_t pixelIndex) const
//{
//    return pixelLocations[pixelIndex];
//}
//
//inline void DirectTracerMiddleCallback::SetSection(const Vector2i& start,
//                                                   const Vector2i& end)
//{
//    portionStart = start;
//    portionEnd = end;
//    portionEnd = Vector2i::Min(portionEnd, resolution);
//
//    // Allocate an image section
//    Vector2i size = portionEnd - portionStart;
//    pixelLocations.resize(size[0] * size[1]);
//}
//
//inline void DirectTracerMiddleCallback::SetResolution(const Vector2i& v)
//{
//    resolution = v;
//}
//
//inline const Vector2i& DirectTracerMiddleCallback::Start() const
//{
//    return portionStart;
//}
//
//inline const Vector2i& DirectTracerMiddleCallback::End() const
//{
//    return portionEnd;
//}
//
//inline const Vector2i& DirectTracerMiddleCallback::Resolution() const
//{
//    return resolution;
//}

static_assert(IsTracerClass<RefPGTracer>::value,
              "RefPGTracer is not a Tracer Class.");
