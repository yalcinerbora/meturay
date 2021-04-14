#pragma once
/**

Visor Interface

Visor is a standalone program to monitor current
image that si being rendered by Tracers

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

struct VisorError;
struct VisorCamera;

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

class VisorI
{
    public:
        virtual                         ~VisorI() = default;

        // Interface
        virtual VisorError              Initialize() = 0;
        virtual bool                    IsOpen() = 0;
        virtual void                    Render() = 0;
        // Input System
        virtual void                    SetInputScheme(VisorInputI&) = 0;
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
        virtual void                    SetCamera(const VisorCamera&) = 0;
};