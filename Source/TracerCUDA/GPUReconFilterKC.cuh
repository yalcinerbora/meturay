#pragma once

#include <cub/cub.cuh>

#include "RayLib/AABB.h"
#include "RayLib/Vector.h"
#include "ImageStructs.h"
#include "CudaSystem.h"

__device__ __host__
inline uint32_t FilterRadiusToPixelWH(float filterRadius)
{
    // At every 0.5 increment conservative pixel estimate is increasing
    // [0]          = Single Pixel (Special Case)
    // (0, 0.5]     = 2x2
    // (0.5, 1]     = 3x3
    // (1, 1.5]     = 4x4
    // (1.5, 2]     = 5x5
    // etc...
    uint32_t result = 1;
    if(filterRadius == 0.0f) return result;

    // Do division
    uint32_t quot = static_cast<uint32_t>(filterRadius / 0.5f);
    float remainder = fmod(filterRadius, 0.5f);

    // Exact divisions reside on previous segment
    if(remainder == 0.0f) quot -= 1;
    result += (quot + 1);
    return result;
}

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
    const int32_t rangeInt = static_cast<int32_t>(FilterRadiusToPixelWH(filterRadius));
    const Vector2i rangeXY = Vector2i(-(rangeInt - 1) / 2,
                                      (rangeInt + 2) / 2);
    // Don't use 1.0f exactly here
    // pixel is [0,1) internally
    const float pixWidth = nextafter(1.0f, 0.0f);
    const float filterRSqr = filterRadius * filterRadius;

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
        Vector2f sampleImgCoords = gImgCoords[threadId];
        Vector2f relImgCoords = Vector2f(modf(sampleImgCoords[0],
                                              &(samplePixId2D[0])),
                                         modf(sampleImgCoords[1],
                                              &(samplePixId2D[1])));
        Vector2i samplePixId2DInt = Vector2i(samplePixId2D);

        // Adjsut the pixel range wrt. pixelCoords
        Vector2i pixRangeX = rangeXY;
        Vector2i pixRangeY = rangeXY;
        // Shift the range window unless radius is odd
        if(rangeInt % 2 == 0)
        {
            if(relImgCoords[0] < 0.5f) pixRangeX -= Vector2i(1);
            if(relImgCoords[1] < 0.5f) pixRangeY -= Vector2i(1);
        }

        //printf("Range X [%d, %d), Y [%d, %d)\n",
        //       pixRangeX[0], pixRangeX[1],
        //       pixRangeY[0], pixRangeY[1]);

        // Actual write
        int writeCounter = 0;
        for(int y = pixRangeY[0]; y < pixRangeY[1]; y++)
        for(int x = pixRangeX[0]; x < pixRangeX[1]; x++)
        {
            // Find the closest point on the pixel
            Vector2f pixCoord = samplePixId2D + Vector2f(static_cast<float>(x),
                                                         static_cast<float>(y));
            Vector2f pixCenter = pixCoord + Vector2f(0.5f);
            //printf("S(%f, %f): Pixel (%f, %f)-(%f, %f) ",
            //       relImgCoords[0], relImgCoords[1],
            //       pixel.Min()[0], pixel.Min()[1],
            //       pixel.Max()[0], pixel.Max()[1]);

            Vector distVec = (sampleImgCoords - pixCenter);
            // Do range check
            // (special case when radius == 0, directly accept)
            if(rangeInt == 1 ||
               distVec.LengthSqr() <= filterRSqr)
            {
                if(writeCounter == maxPixelPerSample)
                    printf("Filter Error: Too many pixels!\n");

                //printf("YES\n");

                // This pixel is affected by this sample
                Vector2i actualPixId = Vector2i(samplePixId2DInt[0] + x,
                                                samplePixId2DInt[1] + y);
                bool pixXInside = (actualPixId[0] >= 0 &&
                                    actualPixId[0] < imgResolution[0]);
                bool pixYInside = (actualPixId[1] >= 0 &&
                                    actualPixId[1] < imgResolution[1]);

                // Only make the sample if pixel is in the actual image
                if(pixXInside && pixYInside)
                {
                    uint32_t pixelId = (actualPixId[1] * imgResolution[0] +
                                        actualPixId[0]);

                    gLocalPixIds[writeCounter] = pixelId;
                    gLocalsampleIndices[writeCounter] = threadId;
                    writeCounter++;
                }
            }
            //else printf("NO\n");
        }
        // Write invalid to the unused locations
        for(int i = writeCounter; i < maxPixelPerSample; i++)
        {
            gLocalPixIds[i] = UINT32_MAX;
            gLocalsampleIndices[i] = UINT32_MAX;
        }
    }
}

template <class T, class Filter, int TPB_X>
__global__ __launch_bounds__(TPB_X)
static void KCFilterToImgBlock(ImageGMem<T> img,
                               // Inputs per block
                               const uint32_t* gOffsets,
                               const uint32_t* gPixelIds,
                               // Inputs per thread
                               const uint32_t* gSampleIds,
                               // Inputs Accessed by SampleId
                               const T* gValues,
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

        // Skip this segment
        if(sPixelId == UINT32_MAX) continue;

        // Calculate Pixel Coordinates on the image space
        const Vector2f pixCoords = PixelIdToImgCoords(sPixelId);
        // Calculate iteration count
        // How many passes this block needs to finish
        const uint32_t elementCount = sOffset[1] - sOffset[0];
        uint32_t iterations = (elementCount + TPB_X - 1) / TPB_X;

        // These variables are only valid on main thread
        float totalWeight = 0.0f;
        T totalValue = T(0.0f);
        // Do the reduction batch by batch
        for(uint32_t i = 0; i < iterations; i++)
        {
            const uint32_t localIndex = localId + i * TPB_X;
            const uint32_t globalSampleIdIndex = sOffset[0] + localIndex;

            // Fetch Value & Image Coords
            T value = T(0.0f);
            float filterWeight = 0.0f;
            if(localIndex < elementCount)
            {
                uint32_t sampleId = gSampleIds[globalSampleIdIndex];
                value = gValues[sampleId];
                Vector2f sampleCoords = gImgCoords[sampleId];
                // Do the filter operation here
                // If this operator unavailable just do
                filterWeight = filter(pixCoords, sampleCoords);
            }
            T weightedVal = filterWeight * value;

            // Now do the reduction operations
            totalWeight += BlockReduceF(tempShared.reduceFMem).Sum(filterWeight);
            __syncthreads();
            totalValue += BlockReduceV4(tempShared.reduceV4Mem).Sum(weightedVal);
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

template <class T, class Filter, int TPB_X>
__global__ __launch_bounds__(TPB_X)
static void KCFilterToImgWarp(ImageGMem<T> img,
                              // Inputs per block
                              const uint32_t* gOffsets,
                              const uint32_t* gPixelIds,
                              // Inputs per thread
                              const uint32_t* gSampleIds,
                              // Inputs Accessed by SampleId
                              const T* gValues,
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

    // Each warp is responsible for a pixel
    // Calculate the indices
    // Warp-stride loop related params
    uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t globalWarpId = globalId / WARP_SIZE;
    uint32_t localWarpId = threadIdx.x / WARP_SIZE;
    uint32_t laneId = globalId % WARP_SIZE;
    uint32_t totalWarps = (gridDim.x * blockDim.x) / WARP_SIZE;

    static constexpr uint32_t WARP_PER_BLOCK = TPB_X / WARP_SIZE;
    // Specialize WarpReduce for actual filter and value
    using WarpReduceF = cub::WarpReduce<float, WARP_SIZE>;
    using WarpReduceV4 = cub::WarpReduce<T, WARP_SIZE>;

    // Shared Memory
    __shared__ struct
    {
        union
        {
            typename WarpReduceF::TempStorage  reduceFMem;
            typename WarpReduceV4::TempStorage reduceV4Mem;

        } warp[WARP_PER_BLOCK];
    } sMem;

    // Grid-Stride Loop (Warp)
    for(uint32_t segmentIndex = globalWarpId; segmentIndex < segmentCount;
        segmentIndex += totalWarps)
    {
        // TODO: Is this better or single thread loads then broadcasts??
        const Vector2ui offset = Vector2ui(gOffsets[segmentIndex],
                                           gOffsets[segmentIndex + 1]);
        const int32_t pixelId = static_cast<int32_t>(gPixelIds[segmentIndex]);

        // Skip this segment if invalid
        if(pixelId == UINT32_MAX) continue;

        // Calculate Pixel Coordinates on the image space
        const Vector2f pixCoords = PixelIdToImgCoords(pixelId);
        // Calculate iteration count
        // How many passes this block needs to finish
        const uint32_t elementCount = offset[1] - offset[0];
        uint32_t iterations = (elementCount + WARP_SIZE - 1) / WARP_SIZE;

        // These variables are only valid on main thread
        float totalWeight = 0.0f;
        T totalValue = T(0.0f);
        // Do the reduction batch by batch
        for(uint32_t i = 0; i < iterations; i++)
        {
            const uint32_t localIndex = laneId + i * WARP_SIZE;
            const uint32_t globalSampleIdIndex = offset[0] + localIndex;

            // Fetch Value & Image Coords
            T value = T(0.0f);
            float filterWeight = 0.0f;
            if(localIndex < elementCount)
            {
                uint32_t sampleId = gSampleIds[globalSampleIdIndex];
                value = gValues[sampleId];
                Vector2f sampleCoords = gImgCoords[sampleId];
                // Do the filter operation here
                // If this operator unavailable just do
                filterWeight = filter(pixCoords, sampleCoords);
            }
            T weightedVal = filterWeight * value;

            // Now do the reduction operations
            totalWeight += WarpReduceF(sMem.warp[localWarpId].reduceFMem).Sum(filterWeight);
            totalValue += WarpReduceV4(sMem.warp[localWarpId].reduceV4Mem).Sum(weightedVal);
        }
        // Now all is reduced, warp leader can write to the img buffer
        if(laneId == 0)
        {
            int32_t segmentIndex = PixelIdToImgSegmentLocalId(pixelId);
            img.gSampleCounts[segmentIndex] += totalWeight;
            img.gPixels[segmentIndex] += totalValue;
        }
    }

}