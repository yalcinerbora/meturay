#pragma once

/**

Tracer Interface

Main Interface for Tracer. Tracer is a integrator like interface (from PBR book)
However it has additional functionality.

First of all this class will be a threaded interface meaning it should implement
callback functionality in order to return "stuff". TracerCallbacksI provides function pointers.
(Most of the time a image segment, but analytic data and errors should also be returned)

The interface is designed specifically for GPU. (Because of that it required a GPUScene Interface)
It is also responsible for utilizing all GPUs on the computer.

*/

#include "Vector.h"
#include "Types.h"
#include "Constants.h"
#include "HitStructs.h"
#include "TracerOptionsI.h"

struct VisorTransform;
struct TracerError;
class TracerCallbacksI;
class GPUCameraI;

class GPUTracerI
{
    public:
        virtual                         ~GPUTracerI() = default;

        // =====================//
        // RESPONSE FROM TRACER //
        // =====================//
        // Callbacks
        virtual void                    AttachTracerCallbacks(TracerCallbacksI&) = 0;

        // ===================//
        // COMMANDS TO TRACER //
        // ===================//
        virtual TracerError             Initialize()  = 0;
        // Option Related
        virtual TracerError             SetOptions(const TracerOptionsI&) = 0;
        virtual void                    SetParameters(const TracerParameters&) = 0;
        virtual void                    AskOptions() = 0;
        virtual void                    AskParameters() = 0;

        // Rendering Related
        // Generate Work for Scene Camera
        virtual void                    GenerateWork(uint32_t cameraId) = 0;
        // Generate Work for Arbitrary Visor Transform
        virtual void                    GenerateWork(const VisorTransform&, uint32_t cameraId) = 0;
        // Generate Work for Arbitrary GPU Camera
        virtual void                    GenerateWork(const GPUCameraI&) = 0;
        virtual bool                    Render() = 0;   // Continue Working (until no work is left)
        virtual void                    Finalize() = 0; // Finalize work (write to image)

        // Image Seated
        virtual void                    SetImagePixelFormat(PixelFormat) = 0;
        virtual void                    ReportionImage(Vector2i start = Zero2i,
                                                       Vector2i end = BaseConstants::IMAGE_MAX_SIZE) = 0;
        virtual void                    ResizeImage(Vector2i resolution) = 0;
        virtual void                    ResetImage() = 0;

        // Misc
        virtual size_t                  TotalGPUMemoryUsed() const = 0;
};