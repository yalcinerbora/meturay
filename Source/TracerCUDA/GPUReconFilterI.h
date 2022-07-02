#pragma once

#include "RayLib/Vector.h"

class ImageMemory;
class CudaSystem;

class GPUReconFilterI
{
    public:
        virtual                         ~GPUReconFilterI() = default;
        // Interface
        virtual const char*             Type() const = 0;
        virtual size_t                  UsedGPUMemory() const = 0;
        // Actual Functionality
        // Filter the samples to the image
        virtual void                    FilterToImg(ImageMemory&,
                                                    const Vector4f* dValues,
                                                    const Vector2f* dImgCoords,
                                                    uint32_t sampleCount,
                                                    const CudaSystem& system) = 0;
};