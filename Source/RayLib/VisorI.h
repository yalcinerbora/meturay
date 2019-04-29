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

struct VisorOptions
{
    // Technical
    size_t              eventBufferSize;

    // Window Related
    bool                stereoOn;
    PixelFormat         iFormat;
    Vector2i            iSize;
};

class VisorI
{
    public:
        virtual                         ~VisorI() = default;

        // Interface
        virtual bool                    IsOpen() = 0;
        virtual void                    Render() = 0;
        virtual void                    ProcessInputs() = 0;
        // Input System
        virtual void                    SetInputScheme(VisorInputI*) = 0;
        virtual void                    SetCallbacks(VisorCallbacksI*) = 0;

        // Data Related
        // Reset Data (Clears the RGB(A) Buffer of the Image)
        // and resets total accumulated rays
        virtual void                    ResetSamples(Vector2i start = Zero2i,
                                                     Vector2i end = BaseConstants::IMAGE_MAX_SIZE) = 0;
        // Append incoming data from
        virtual void                    AccumulatePortion(const std::vector<Byte> data,
                                                          PixelFormat, int sampleCount,
                                                          Vector2i start = Zero2i,
                                                          Vector2i end = BaseConstants::IMAGE_MAX_SIZE) = 0;
        // Options
        virtual const VisorOptions&     VisorOpts() const = 0;
        // Misc
        virtual void                    SetWindowSize(const Vector2i& size) = 0;
        virtual void                    SetFPSLimit(float) = 0;

};