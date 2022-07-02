#pragma once

#include <cub/cub.cuh>

#include "RayLib/Vector.h"
#include "ImageStructs.h"
#include "CudaSystem.h"

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCExpandSamplesToPixels(// Outputs
                                    uint32_t* gPixelIds,
                                    uint32_t* gSampelIndices,
                                    // Inputs
                                    const Vector2f* gImgCoords,
                                    // Constants
                                    uint32_t maxPixelPerSample,
                                    uint32_t totalSampleCount,
                                    float filterRadius,
                                    Vector2i imgResolution)
{
    // Conservative range of pixels
    const uint32_t rangeInt = static_cast<int>(ceil(filterRadius));
    const Vector2i rangeXY = Vector2i(-(rangeInt * 2), rangeInt * 2);

    // Grid Stride Loop
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalSampleCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint32_t* gLocalPixIds = gPixelIds + maxPixelPerSample * threadId;
        uint32_t* gLocalsampleIndices = gSampelIndices + maxPixelPerSample * threadId;

        // TODO: Change the parallel logic to
        // One warp per sample here
        // Currently it is one thread per sample
        // (It may be faster?)
        // Load img coordinates for this sample
        Vector2f samplePixId2D;
        Vector2f imgCoords = gImgCoords[threadId];
        Vector2f relImgCoords = Vector2f(modf(imgCoords[0],
                                              &(samplePixId2D[0])),
                                         modf(imgCoords[1],
                                              &(samplePixId2D[1])));
        Vector2i samplePixId2DInt = Vector2i(samplePixId2D);

        // Determine Coalesced loop size
        uint32_t totalRange = rangeXY[0] + rangeXY[1] + 1;
        totalRange *= totalRange;

        int writeCounter = 0;
        for(int y = rangeXY[0]; y <= rangeXY[1]; y++)
        for(int x = rangeXY[0]; x <= rangeXY[1]; x++)
        {
            // Find the closest point on the pixel
            Vector2f pixCoord = Vector2f(static_cast<float>(x),
                                        static_cast<float>(y));
            pixCoord += Vector2f((x < 0) ? 1.0f : 0.0f,
                                 (y < 0) ? 1.0f : 0.0f);
            //
            float dist = (pixCoord - relImgCoords).LengthSqr();
            if(dist < filterRadius * filterRadius)
            {
                if(writeCounter == maxPixelPerSample)
                    printf("Filter Error: Too many pixels!\n");

                // This pixel is affected by this sample
                uint32_t pixelId = ((samplePixId2DInt[1] + y) * imgResolution[0] +
                                    samplePixId2DInt[0] + x);

                gLocalPixIds[writeCounter] = pixelId;
                gLocalsampleIndices[writeCounter] = threadId;
                writeCounter++;
            }
        }
    }
}

template <class T, class Filter, int TPB_X>
__global__ __launch_bounds__(TPB_X)
static void KCFilterToImg(ImageGMem<T> img,
                          // Inputs per block
                          const uint32_t* gOffsets,
                          const uint32_t* gPixelIds,
                          // Inputs per thread
                          const uint32_t* gSampleIds,
                          // Inputs Accessed by SampleId
                          const Vector4f* gValues,
                          const Vector2f* gImgCoords,
                          // Constants
                          Vector2i imgSegmentSize,
                          Vector2i imgSegmentOffset,
                          Vector2i imgResolution,
                          Filter filter,
                          uint32_t segmentCount)
{
    auto PixelIdToImgCoords = [&](int32_t pixId) -> Vector2f
    {
        Vector2i pix2D = Vector2i(pixId % imgResolution[0],
                                  pixId / imgResolution[0]);

        return Vector2f(static_cast<float>(pix2D[0]) + 0.5f,
                        static_cast<float>(pix2D[1]) + 0.5f);
    };
    auto PixelIdToImgSegmentLocalId = [&](int32_t pixId) -> int32_t
    {
        Vector2i pix2D = Vector2i(pixId % imgResolution[0],
                                  pixId / imgResolution[0]);
        pix2D -= imgSegmentOffset;
        int32_t segmentPixId = (pix2D[1] * imgSegmentSize[0] +
                                pix2D[0]);

        return segmentPixId;
    };

    const uint32_t localId = threadIdx.x;

    // Specialize BlockReduce for actual filter and value
    using BlockReduceF = cub::BlockReduce<float, TPB_X>;
    using BlockReduceV4 = cub::BlockReduce<T, TPB_X>;

    // Shared Memory
    __shared__ union
    {
        typename BlockReduceF::TempStorage  reduceFMem;
        typename BlockReduceV4::TempStorage reduceV4Mem;
    } tempShared;
    __shared__ Vector2ui sOffset;
    __shared__ int32_t sPixelId;

    // Grid-Stride Loop
    for(uint32_t segmentIndex = blockIdx.x; segmentIndex < segmentCount;
        segmentIndex += gridDim.x)
    {
        // Segment Related Info
        if(localId == 0)
        {
            sPixelId = static_cast<int32_t>(gPixelIds[segmentIndex]);
            sOffset[0] = gOffsets[segmentIndex];
            sOffset[1] = gOffsets[segmentIndex + 1];
        }
        __syncthreads();

        // Calculate Pixel Coordinates on the image space
        const Vector2f pixCoords = PixelIdToImgCoords(sPixelId);
        // Calculate iteration count
        // How many passes this block needs to finish
        const uint32_t elementCount = sOffset[1] - sOffset[0];
        uint32_t iterations = (elementCount + TPB_X - 1) / TPB_X;

        // These variables are only valid on main thread
        float totalWeight = 0.0f;
        T totalValue = Zero4f;
        // Do the reduction batch by batch
        for(uint32_t i = 0; i < iterations; i++)
        {
            uint32_t fetchIndex = sOffset[0] + localId + i * TPB_X;

            // Fetch Value & Image Coords
            T value = T(0.0f);
            Vector2f sampleCoords = Vector2f(FLT_MAX, FLT_MAX);
            if(fetchIndex < elementCount)
            {
                uint32_t sampleId = gSampleIds[fetchIndex];
                value = gValues[sampleId];
                sampleCoords = gImgCoords[sampleId];
            }
            // Do the filter operation & multiply with value
            float filterWeight = filter(pixCoords, sampleCoords);
            T weightedVal = filterWeight * value;

            // Now do the reduction operations
            totalWeight += BlockReduceF(tempShared.reduceFMem).Sum(filterWeight);
            __syncthreads();
            totalValue += BlockReduceV4(tempShared.reduceV4Mem).Sum(totalValue);
        }
        // Now all is reduced, main thread can write to the img buffer
        if(localId == 0)
        {
            int32_t segmentIndex = PixelIdToImgSegmentLocalId(sPixelId);
            img.gSampleCounts[segmentIndex] += totalWeight;
            img.gPixels[segmentIndex] += totalValue;
        }
        // Do sync here, maybe some trailing threads may fetch segment related info
        __syncthreads();
    }
}