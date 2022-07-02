#pragma once

#include "GPUReconFilterI.h"
#include "GPUReconFilterKC.cuh"

#include "DeviceMemory.h"
#include "ImageMemory.h"
#include "RayLib/Options.h"
#include "RayLib/TracerError.h"

#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "ParallelPartition.cuh"

template <class GPUFilterFunctor>
class GPUReconFilter : public GPUReconFilterI
{
    private:
        DeviceMemory            filterMemory;

    protected:
        // Common Json Names
        static constexpr const char* RADIUS_NAME = "radius";

        GPUFilterFunctor        filterGPU;
        float                   filterRadius;

        uint32_t                ConservativePixelPerSample() const;

        // Constructors
                                GPUReconFilter(Options filterOptions,
                                               GPUFilterFunctor f);
    public:
        // Destructor

                                ~GPUReconFilter() = default;

        size_t                  UsedGPUMemory() const override;
        // Actual Functionality
        // Filter the samples to the image
        void                    FilterToImg(ImageMemory&,
                                            const Vector4f* dValues,
                                            const Vector2f* dImgCoords,
                                            uint32_t sampleCount,
                                            const CudaSystem&) override;
};

class GPUBoxFilter
{
    private:
    float       radius;

    public:
    // Constructors & Destructor
                BoxFilter(float radius);
                ~BoxFilter() = default;

    __device__ __host__
    float       operator()(const Vector2f& pixCoord,
                           const Vector2f& sampleCoord) const;
};

class GPUReconFilterBox : public GPUReconFilter<BoxFilter>
{
    private:
    protected:
    public:
        // Constructors & Destructor
                    GPUReconFilter(Options filterOptions);
                    ~GPUReconFilter() = default;
};

inline ReconBoxFilterFunctor::ReconBoxFilterFunctor(float radius)
    : radius(radius)
{}

__device__ inline
float ReconBoxFilterFunctor::operator()(const Vector2f& pixCoord,
                                        const Vector2f& sampleCoord) const
{
    return ((pixCoord - sampleCoord).Length() < radius) ? 1.0f : 0.0f;
}


template <class T>
uint32_t GPUReconFilter<T>::ConservativePixelPerSample() const
{
    uint32_t rangeInt = static_cast<int>(std::ceil(filterRadius));
    // Special case, for zero radius directly write to a pixel
    if(rangeInt == 0) return 1;
    return rangeInt * rangeInt * 4;
}

template <class T>
GPUReconFilter<T>::GPUReconFilter(Options filterOptions, T f)
    : filterGPU(f)
{
    TracerError e = TracerError::OK;
    if((e = filterOptions.GetFloat(filterRadius, RADIUS_NAME)) != TracerError::OK)
        throw TracerException(e);
}

template <class T>
size_t GPUReconFilter<T>::UsedGPUMemory() const
{
    return filterMemory.Size();
}

template <class T>
void GPUReconFilter<T>::FilterToImg(ImageMemory& img,
                                    const Vector4f* dValues,
                                    const Vector2f* dImgCoords,
                                    uint32_t sampleCount,
                                    const CudaSystem& system)
{
    const auto& gpu = system.BestGPU();

    uint32_t pps = ConservativePixelPerSample();
    uint32_t ppsTotal = pps * sampleCount;

    // Required Memory Size
    uint32_t* dPixelIdBuffer0 = nullptr;
    uint32_t* dPixelIdBuffer1 = nullptr;
    uint32_t* dSampleIndicesBuffer0 = nullptr;
    uint32_t* dSampleIndicesBuffer1 = nullptr;
    Byte* dTempMemory = nullptr;

    cub::DoubleBuffer<uint32_t> dbPixIds(dPixelIdBuffer0, dPixelIdBuffer1);
    cub::DoubleBuffer<uint32_t> dbSampleIndices(dSampleIndicesBuffer0, dSampleIndicesBuffer1);

    // After determining pps pairs, we need to do sort these by their pixel
    // We need cub:radix sort temp memory,
    size_t sortTempBufferSize;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sortTempBufferSize,
                                               dbPixIds, dbSampleIndices,
                                               static_cast<int>(ppsTotal)));
    // We need to mark the differences, use "if" operation to find the splits
    size_t ifTempBufferSize;
    CUDA_CHECK(cub::DeviceSelect::If(nullptr, ifTempBufferSize,
                                     dPixelIdBuffer0,
                                     dPixelIdBuffer0,
                                     dPixelIdBuffer0,
                                     static_cast<int>(ppsTotal),
                                     ValidSplit()));
    size_t tempMemSize = std::max(ifTempBufferSize, sortTempBufferSize);


    // Now we can do the allocation
    GPUMemFuncs::AllocateMultiData(std::tie(dPixelIdBuffer0,
                                            dPixelIdBuffer1,
                                            dSampleIndicesBuffer0,
                                            dSampleIndicesBuffer1,
                                            dTempMemory),
                                   filterMemory,
                                   {ppsTotal, ppsTotal,
                                   ppsTotal, ppsTotal,
                                   tempMemSize});


    // Call the PPS Generation Kernel
    gpu.GridStrideKC_X(0, (cudaStream_t)0, sampleCount,
                       //
                       KCExpandSamplesToPixels,
                       // Outputs
                       dPixelIdBuffer0,
                       dSampleIndicesBuffer0,
                       // Inputs
                       dImgCoords,
                       //
                       pps,
                       sampleCount,
                       filterRadius,
                       img.Resolution());
    // Do the sort
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(static_cast<void*>(dTempMemory), sortTempBufferSize,
                                               dbPixIds, dbSampleIndices,
                                               static_cast<int>(ppsTotal)));


    // Rename the buffers for if operations etc.
    uint32_t* dSortedPixelIds;
    uint32_t* dSortedSampleIndices;
    uint32_t* dIfOutput; uint32_t* dIfInput;
    if(dPixelIdBuffer0 == dbPixIds.Current())
    {
        dSortedPixelIds = dPixelIdBuffer0;
        dIfOutput = dPixelIdBuffer1;
    }
    else
    {
        dSortedPixelIds = dPixelIdBuffer1;
        dIfOutput = dPixelIdBuffer0;
    }
    if(dSampleIndicesBuffer0 == dbSampleIndices.Current())
    {
        dSortedSampleIndices = dSampleIndicesBuffer0;
        dIfInput = dSampleIndicesBuffer1;
    }
    else
    {
        dSortedSampleIndices = dSampleIndicesBuffer1;
        dIfInput = dSampleIndicesBuffer0;
    }

    // Mark the splits
    uint32_t markLocCount = ppsTotal - 1;
    gpu.GridStrideKC_X(0, (cudaStream_t)0, markLocCount,
                       //
                       KCMarkSplits<uint32_t>,
                       //
                       dIfInput,
                       dSortedPixelIds,
                       markLocCount);


    // Densely pack the mark locations using cub "if" function
    uint32_t* dSplitCount = static_cast<uint32_t*>(dIfOutput);
    uint32_t* dDenseSplitIndices = static_cast<uint32_t*>(dIfOutput) + 1;
    CUDA_CHECK(cub::DeviceSelect::If(static_cast<void*>(dTempMemory), ifTempBufferSize,
                                     dIfInput,
                                     dDenseSplitIndices,
                                     dSplitCount,
                                     static_cast<int>(ppsTotal),
                                     ValidSplit()));
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)0));

    // Load the dense size
    uint32_t hSelectCount;
    CUDA_CHECK(cudaMemcpy(&hSelectCount, dSplitCount,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));


    uint32_t* dDenseKeys = dIfInput;
    gpu.GridStrideKC_X(0, (cudaStream_t)0, markLocCount,
                       //
                       KCFindSplitBatches<uint32_t>,
                       //
                       dDenseKeys,
                       dDenseSplitIndices,
                       dSortedPixelIds,
                       hSelectCount);

    const uint32_t* dOffsets = dDenseSplitIndices;
    const uint32_t* dPixelIds = dDenseKeys;
    const uint32_t* dSampleIds = dSortedSampleIndices;
    uint32_t segmentCount = hSelectCount;

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