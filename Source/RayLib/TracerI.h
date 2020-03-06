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

struct TracerError;
struct TracerOptions;
struct CameraPerspective;

class GPUSceneI;

// Main Tracer Logicc
class TracerCallbacksI;
class TracerBaseLogicI;

class TracerGPUI
{
    public:
        virtual                         ~TracerNodeI() = default;

        // =====================//
        // RESPONSE FROM TRACER //
        // =====================//
        // Callbacks
        virtual void                    AttachTracerCallbacks(TracerCallbacksI&) = 0;

        // ===================//
        // COMMANDS TO TRACER //
        // ===================//
        // Main Calls
        virtual TracerError             Initialize()  = 0;

        // Rendering Related
        virtual void                    GenerateWork(const GPUSceneI& scene,
                                                     int cameraId) = 0;
        virtual void                    GenerateWork(const GPUSceneI& scene,
                                                     const CameraPerspective&) = 0;
        virtual bool                    Render() = 0;   // Continue Working (untill no work is left)
        virtual void                    Finalize() = 0; // Finalize work (write to image)

        // Image Reated
        virtual void                    SetImagePixelFormat(PixelFormat) = 0;
        virtual void                    ReportionImage(Vector2i start = Zero2i,
                                                       Vector2i end = BaseConstants::IMAGE_MAX_SIZE) = 0;
        virtual void                    ResizeImage(Vector2i resolution) = 0;
        virtual void                    ResetImage() = 0;
};