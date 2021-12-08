#pragma once
/**

Visor Interface

Visor is a standalone program to monitor current
image that is being rendered by Tracers

VisorView Interface encapsulates rendering window and real-time GPU portion
Visor

*/

#include <cstddef>
#include <vector>
#include "Types.h"
#include "Vector.h"
#include "Constants.h"

class VisorInputI;
class VisorCallbacksI;

struct VisorTransform;
struct VisorError;

class TracerOptions;
struct TracerParameters;

struct TracerAnalyticData;
struct SceneAnalyticData;

class MovementSchemeI;
using MovementSchemeList = std::vector<std::unique_ptr<MovementSchemeI>>;

struct VisorOptions
{
    // Technical
    size_t              eventBufferSize;

    // Window Related
    bool                stereoOn;
    bool                vSyncOn;
    PixelFormat         wFormat;
    Vector2i            wSize;
    float               fpsLimit;
    // Misc
    bool                enableGUI;
    bool                enableTMO;
};

class ImageSaverI
{
    public:
        virtual         ~ImageSaverI() = default;
        // Image Save Related
        virtual void    SaveImage(bool saveAsHDR) = 0;
};

class VisorI : public ImageSaverI
{
    public:
        virtual                         ~VisorI() = default;

        // Interface
        virtual VisorError              Initialize(VisorCallbacksI&,
                                                   const KeyboardKeyBindings&,
                                                   const MouseKeyBindings&,
                                                   MovementSchemeList&&) = 0;
        virtual bool                    IsOpen() = 0;
        virtual void                    Render() = 0;
        // Data Related
        // Set the resolution of the rendering data
        virtual void                    SetImageRes(Vector2i resolution) = 0;
        virtual void                    SetImageFormat(PixelFormat f) = 0;
        // Reset Data (Clears the RGB(A) Buffer of the Image)
        // and resets total accumulated rays
        virtual void                    ResetSamples(Vector2i start = Zero2i,
                                                     Vector2i end = BaseConstants::IMAGE_MAX_SIZE) = 0;
        // Append incoming data from
        virtual void                    AccumulatePortion(const std::vector<Byte> data,
                                                          PixelFormat, size_t offset,
                                                          Vector2i start = Zero2i,
                                                          Vector2i end = BaseConstants::IMAGE_MAX_SIZE) = 0;
        // Options
        virtual const VisorOptions&     VisorOpts() const = 0;
        // Misc
        virtual void                    SetWindowSize(const Vector2i& size) = 0;
        virtual void                    SetFPSLimit(float) = 0;
        virtual Vector2i                MonitorResolution() const = 0;
        // Setting/Releasing rendering context on current thread
        virtual void                    SetRenderingContextCurrent() = 0;
        virtual void                    ReleaseRenderingContext() = 0;
        // Main Thread only Calls
        virtual void                    ProcessInputs() = 0;
        // Updates (Data coming from Tracer Callbacks)
        virtual void                    Update(const VisorTransform&) = 0;
        virtual void                    Update(uint32_t sceneCameraCount) = 0;
        virtual void                    Update(const SceneAnalyticData&) = 0;
        virtual void                    Update(const TracerAnalyticData&) = 0;
        virtual void                    Update(const TracerOptions&) = 0;
        virtual void                    Update(const TracerParameters&) = 0;
};