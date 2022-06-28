#pragma once

#include "RayLib/Vector.h"
#include "DeviceMemory.h"
#include "ImageMemory.h"

struct ImageSampleGMem
{
    Vector3f*   value;
    Vector2f*   pixCoords;
};

struct ImageSampleGMemConst
{
    const Vector3f*     value;
    const Vector2f*     pixCoords;
};

class GPUReconFilterI
{
    private:
    public:
        virtual         ~GPUReconFilterI() = default;

        // Interface
        virtual const char*             Type() const = 0;
        // Basic Getters
        virtual size_t                  SampleSize() const = 0;
        virtual size_t                  UsedGPUMemory() const = 0;
        // Sample Buffer GMem
        virtual ImageSampleGMem         SamplesGMem() = 0;
        virtual ImageSampleGMemConst    SamplesGMem() const = 0;

        // Actual Functionality
        // Filter the samples to the image
        //template <class T>
        //void                            FilterToImg(ImageGMem<T>);

        virtual void                    ResizeSampleBuffer(size_t newSampleCount);


};