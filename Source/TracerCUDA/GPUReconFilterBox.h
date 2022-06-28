#pragma once

#include "GPUReconFilterI.h"
#include "GPUReconFilterKC.cuh"

#include "DeviceMemory.h"
#include "ImageMemory.h"
#include "RayLib/Options.h"
#include "RayLib/TracerError.h"

#include "CudaSystem.h"
#include "CudaSystem.hpp"

class ReconBoxFilterFunctor
{
    private:
        float       radius;

    public:
        // Constructors & Destructor
                    ReconBoxFilterFunctor(float radius);
                    ~ReconBoxFilterFunctor() = default;

        __device__
        float       operator()(const Vector2f& pixCoord,
                               const Vector2f& sampleCoord) const;
};

template <class GPUFilterFunctor>
class GPUReconFilter : public GPUReconFilterI
{
    private:
        DeviceMemory            mem;

    protected:
        // Common Json Names
        static constexpr const char* RADIUS_NAME = "radius";

        GPUFilterFunctor        filterGPU;
        float                   filterRadius;

    public:
        // Constructors & Destructor
                                GPUReconFilter(Options filterOptions);

        size_t                  UsedGPUMemory() const override;
        // Actual Functionality
        // Filter the samples to the image
        void                    FilterToImg(ImageMemory&,
                                            const Vector4f* dValues,
                                            const Vector2f* dImgCoords,
                                            size_t sampleCount,
                                            const CudaSystem&) override;
};

ReconBoxFilterFunctor::ReconBoxFilterFunctor(float radius)
    : radius(radius)
{}

__device__
float ReconBoxFilterFunctor::operator()(const Vector2f& pixCoord,
                                        const Vector2f& sampleCoord) const
{
    return ((pixCoord - sampleCoord).Length() < radius) ? 1.0f : 0.0f;
}

template <class T>
GPUReconFilter<T>::GPUReconFilter(Options filterOptions)
{
    TracerError e = TracerError::OK;
    if((e = filterOptions.GetFloat(filterRadius, RADIUS_NAME)) != TracerError::OK)
        throw TracerException(e);
}

template <class T>
void GPUReconFilter<T>::ResizeMemory(size_t sampleCount)
{

}

template <class T>
size_t GPUReconFilter<T>::UsedGPUMemory() const
{
    return mem.Size();
}

template <class T>
void GPUReconFilter<T>::FilterToImg(ImageMemory& img,
                                    const Vector4f* dValues,
                                    const Vector2f* dImgCoords,
                                    size_t sampleCount,
                                    const CudaSystem& system)
{
    const auto& gpu = system.BestGPU();


    gpu.GridStrideKC_X();


    // Find out how many pixels a sample affects.
    // Using that information generate all temp buffers

    // Store the affected pixel count per sample into a buffer

    // Scan it to find offsets

    // Write the pixelIds / sampleIndex pairs
    // into a temp buffer,

    // Sort it wrt. pixelId

    // Now we have a buffer that a kernel can act up on wrt. to pixel

    //

    // Partition this buffer
    // Inputs per block


    uint32_t* dOffsets;
    int32_t* dPixelIds;
    const uint32_t* dSampleIds;
    size_t segmentCount = 0;

    // Call segmented reduce
    static constexpr uint32_t TPB_X = 128;
    const uint32_t filterKernelBlockCount = (gpu.MaxActiveBlockPerSM(TPB_X) *
                                             gpu.SMCount() * TPB_X) / TPB_X;

    gpu.ExactKC_X(0, (cudaStream_t)0,
                  TPB_X, filterKernelBlockCount,
                  //
                  KCFilterToImg<Vector4f, T, TPB_X>,
                  // Out
                  img.GMem<Vector4f>(),
                  // In
                  dOffsets,
                  dPixelIds,
                  dSampleIds,
                  dValues,
                  dImgCoords,
                  // Constants
                  img.SegmentSize(),
                  img.SegmentOffset(),
                  img.Resolution(),

                  filterGPU,
                  segmentCount);
}