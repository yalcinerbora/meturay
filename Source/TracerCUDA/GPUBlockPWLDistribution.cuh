#pragma once

#include "cub/block/block_scan.cuh"

#include "RayLib/Vector.h"
#include "RayLib/HybridFunctions.h"

#include "BlockSegmentedScan.cuh"
#include "RNGenerator.h"
#include "BinarySearch.cuh"

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
class BlockPWLDistribution2D
{
    private:
    static constexpr bool TPBCheck()
    {
        auto PIX_COUNT = (X * Y);
        if(TPB > PIX_COUNT) return TPB % PIX_COUNT == 0;
        if(TPB <= PIX_COUNT) return PIX_COUNT % TPB == 0;
        return false;
    }

    // No SFINAE, just static assert
    static_assert(TPBCheck(),
                  "TBP and (X * Y) must be divisible, (X*Y) / TBP or TBP / (X*Y)");

    static constexpr float X_FLOAT          = static_cast<float>(X);
    static constexpr float Y_FLOAT          = static_cast<float>(Y);
    static constexpr float DELTA_X          = 1.0f / (X_FLOAT - 1.0f);
    static constexpr float DELTA_Y          = 1.0f / (Y_FLOAT - 1.0f);

    using BlockSScan    = BlockSegmentedScan<float, TPB, X>;
    using BlockScan     = cub::BlockScan<float, TPB>;
    // TODO: This does not work, probably ask it on the forums?
    // "using BlockScan = cub::BlockScan<float, Y>;"

    public:
    static constexpr uint32_t DATA_PER_THREAD   = std::max(1u, (X * Y) / TPB);
    static constexpr uint32_t PIX_COUNT         = (X * Y);

    struct TempStorage
    {
        union
        {
            typename BlockSScan::TempStorage    sSScanTempStorage;
            typename BlockScan::TempStorage     sScanTempStorage;
        } algo;
        float sCDFX[Y][X];
        float sPDFX[Y][X];
        // Y axis distribution data (PDF of CDF depending on the situation)
        float sCDFY[Y];
        float sPDFY[Y];
    };

    private:
    TempStorage&    sMem;
    const uint32_t  threadId;
    const bool      isColumnThread;
    const bool      isMainThread;
    const bool      isRowLeader;
    const bool      isValidThread;

    __device__
    static float    PDFRegion(float a, float b, float u);
    __device__
    static float    SampleRegion(float& newU, float a, float b, float index);

    protected:
    public:
    // Constructors & Destructor
    __device__
                    BlockPWLDistribution2D(TempStorage& storage,
                                           const float(&data)[DATA_PER_THREAD]);

    template <class RNG>
    __device__
    Vector2f        Sample(float& pdf, Vector2f& index, RNG& rng) const;

    __device__
    float           Pdf(const Vector2f& index) const;

    __device__
    void            DumpSharedMem(float* funcXOut, float* cdfXOut,
                                  float* funcYOut, float* cdfYOut) const;
};

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__
float BlockPWLDistribution2D<TPB, X, Y>::PDFRegion(float a, float b, float u)
{
    // Ray Tracing Gems I: Chapter 16 "Sampling Transformations Zoo"
    return HybridFuncs::Lerp(a, b, u);
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__
float BlockPWLDistribution2D<TPB, X, Y>::SampleRegion(float& newOffset, float a, float b, float index)
{
    // split the index to integral-fraction (we will remap fraction and return it)
    float integralPart = floorf(index);
    float u = index - integralPart;
    // Ray Tracing Gems I: Chapter 16 "Sampling Transformations Zoo"
    // Refactor "u" if this patch is not linear (CDF is quadratic)
    if(a != b)
    {
        u = (a - sqrtf(HybridFuncs::Lerp(a * a, b * b, u))) / (a - b);
        u = HybridFuncs::Clamp(u, 0.0f, 1.0f);
    }
    newOffset = u;
    return integralPart + u;
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__
BlockPWLDistribution2D<TPB, X, Y>::BlockPWLDistribution2D(TempStorage& storage,
                                                          const float(&data)[DATA_PER_THREAD])
    : sMem(storage)
    , threadId(threadIdx.x)
    , isColumnThread(threadIdx.x < Y)
    , isMainThread(threadIdx.x == 0)
    , isRowLeader((threadIdx.x % X) == 0)
    , isValidThread(threadIdx.x < PIX_COUNT)
{
    // We need the neighboring data for integral calculation.
    // (Piecewise-Linear function segment integral is a trapezoid area)
    // Unlike PWC load the function first then threads can access neighboring,
    // data
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        //  Determine your row
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        uint32_t columnId = pixelId % X;
        // Maybe there is more threads than rows
        if(rowId < Y)
        {
            sMem.sPDFX[rowId][columnId] = data[i];
        }
    }
    __syncthreads();

    // Trapezoid area lambda
    // TODO: Make this a function later
    auto TrapezoidArea = [](float a, float b, float h)
    {
        return (a + b) * h * 0.5f;
    };

    // Now we can calculate the trapezoid areas
    // DATA_PER_THREAD implicitly defines how many rows
    // we can calculate in parallel
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        //  Determine your row
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        uint32_t columnId = pixelId % X;

        // No range check is needed it is checked in compile-time
        // This class is highly restrictive (TPB should be divisible by X or vice versa)
        // But we may have more rows than "(TPB / X) > Y"
        float functionValue = 0.0f;
        float localIntegral = 0.0f;
        if(rowId < Y)
        {
            functionValue = sMem.sPDFX[rowId][columnId];
            // We have only X-1 area calculations on each row
            // Row leader does not calculate here
            localIntegral = (columnId == 0) ? 0 : TrapezoidArea(sMem.sPDFX[rowId][columnId - 1],
                                                                functionValue,
                                                                DELTA_X);
        }

        // Now we can scan and find the CDF (not a CDF currently
        // we need to divide the actual integral result)
        // Block Segmented Scan (entire block does multi row scan)
        // in parallel
        float cdfData;
        float totalSum;
        BlockSScan(sMem.algo.sSScanTempStorage).InclusiveSum(cdfData, totalSum,
                                                             localIntegral, 0.0f);
        __syncthreads();

        // Row leader stores the integral value for Row PWL Distribution
        // Temporarily use the PDFY array, we will divide it later
        if(isRowLeader && rowId < Y) sMem.sPDFY[rowId] = totalSum;

        // Now to the PDF and CDF normalization
        // and these arrays become the actual PDF, CDF
        float pdfData = functionValue;
        if(totalSum != 0.0f)
        {
            // If total sum is zero
            // meaning that this row is not "probable"
            // Prevent NaN here
            cdfData *= (1.0f / totalSum);
            pdfData *= (1.0f / totalSum);
        }
        // Now store
        if(rowId < Y)
        {
            sMem.sCDFX[rowId][columnId] = cdfData;
            sMem.sPDFX[rowId][columnId] = pdfData;
        }

    }
    // Make sure "sMem.sPDFY" array is fully loaded
    __syncthreads();

    // Generate PDF CDF of the column (marginal pdf)
    //"sPDFY" variable holds the Y values
    float functionValue = 0.0f;
    float localIntegral = 0.0f;
    if(isColumnThread)
    {
        uint32_t columnId = threadId;
        functionValue = sMem.sPDFY[columnId];
        localIntegral = (columnId == 0) ? 0 : TrapezoidArea(sMem.sPDFY[columnId - 1],
                                                            functionValue, DELTA_Y);

    }
    // Get out of divergent scope here to prevent deadlocks
    // Scan operation to generate CDF
    float cdfDataY;
    float totalSum;
    BlockScan(sMem.algo.sScanTempStorage).InclusiveSum(localIntegral, cdfDataY, totalSum);
    __syncthreads();
    // Do the normalization for PDF and CDF
    float pdfDataY = functionValue;
    if(totalSum != 0.0f)
    {
        // If total sum is zero
        // meaning that this row is not probable
        // Prevent NaN here
        cdfDataY *= (1.0f / totalSum);
        pdfDataY *= (1.0f / totalSum);
    }
    else if(isMainThread) printf("Entire 2D Histogram is empty!\n");
    // And write
    if(isColumnThread)
    {
        sMem.sCDFY[threadId] = cdfDataY;
        sMem.sPDFY[threadId] = pdfDataY;
    }
    __syncthreads();
    // All CDF's and PDFs are generated
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
template <class RNG>
__device__ inline
Vector2f BlockPWLDistribution2D<TPB, X, Y>::Sample<RNG>(float& pdf, Vector2f& index,
                                                        RNG& rng) const
{
    static constexpr int32_t CDF_SIZE_Y = Y;
    static constexpr int32_t CDF_SIZE_X = X;
    Vector2f xi = rng.Uniform2D();
    Vector2f uvInner = Zero2f;
    // If entire distribution is invalid
    // Just sample uniformly
    if(sMem.sCDFY[Y - 1] == 0.0f)
    {
        index = xi * Vector2f(X, Y);
        pdf = 1.0f;
        return xi;
    }
    GPUFunctions::BinarySearchInBetween<float>(index[1], xi[1],
                                               sMem.sCDFY, CDF_SIZE_Y);
    int32_t indexYInt = static_cast<int32_t>(index[1]);

    // Extremely rarely index becomes the light count
    // although Uniform should return [0, 1)
    // it still happens due to fp error i guess?
    // if it happens just return the last light on the list
    if(indexYInt >= Y)
    {
        KERNEL_DEBUG_LOG("CUDA Error: Illegal Index on PwL Sample [Y = %f]\n",
                         index[1]);
        indexYInt--;
    }
    // Sample the region
    index[1] = SampleRegion(uvInner[1], sMem.sPDFY[indexYInt], sMem.sPDFY[indexYInt + 1], index[1]);

    // Fetch row and do it again
    const float* sRowCDF = sMem.sCDFX[indexYInt];
    const float* sRowPDF = sMem.sPDFX[indexYInt];
    const float* sRowPDFNext = sMem.sPDFX[indexYInt + 1];

    GPUFunctions::BinarySearchInBetween<float>(index[0], xi[0],
                                               sRowCDF, CDF_SIZE_X);
    int32_t indexXInt = static_cast<int32_t>(index[0]);
    if(indexXInt >= X)
    {
        KERNEL_DEBUG_LOG("CUDA Error: Illegal Index on PwL Sample [X = %f]\n",
                         index[0]);
        indexXInt--;
    }
    // TODO: Should we do multiple sample regions here?
    index[0] = SampleRegion(uvInner[0], sRowPDF[indexXInt], sRowPDF[indexXInt + 1], index[0]);

    // Calculate PDF
    // Bilerp Interpolation of the multiplied PDFs
    using namespace HybridFuncs;
    pdf = Lerp(sMem.sPDFY[indexYInt + 0] * Lerp(sRowPDF[indexXInt], sRowPDF[indexXInt + 1], uvInner[0]),
               sMem.sPDFY[indexYInt + 1] * Lerp(sRowPDFNext[indexXInt], sRowPDFNext[indexXInt + 1], uvInner[0]),
               uvInner[1]);

    //if(pdfY == 0.0f || pdfX == 0.0f)
    //{
    //    printf("[Z] pdf(%.10f, %.10f), xi (%.10f, %.10f), index (%.10f, %.10f) (%d, %d)\n",
    //           pdfX, pdfY, xi[0], xi[1], index[0], index[1],
    //           indexXInt, indexYInt);
    //}
    //if(isnan(pdfY) || isnan(pdfX))
    //{
    //    printf("[NaN] pdf(%.10f, %.10f), xi (%.10f, %.10f), index (%.10f, %.10f) (%d, %d)\n",
    //           pdfX, pdfY, xi[0], xi[1], index[0], index[1],
    //           indexXInt, indexYInt);
    //}
    //if(index.HasNaN())
    //{
    //    printf("[NaN] index(%f, %f)\n", index[0], index[1]);
    //}

    // Return the index as a normalized coordinate as well
    return index * Vector2f(DELTA_X, DELTA_Y);
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__ inline
float BlockPWLDistribution2D<TPB, X, Y>::Pdf(const Vector2f& index) const
{
    Vector2ui indexInt = Vector2ui(static_cast<uint32_t>(index[0]),
                                   static_cast<uint32_t>(index[1]));
    Vector2f w = Vector2f(index[0] - truncf(index[0]),
                          index[1] - truncf(index[1]));

    const float* sRowPDF = sMem.sPDFX[indexInt[1]];
    float pdfY = PDFRegion(sMem.sPDFY[indexInt[1]],
                           sMem.sPDFY[indexInt[1] + 1],
                           w[1]);

    float pdfX = PDFRegion(sRowPDF[indexInt[0]],
                           sRowPDF[indexInt[0] + 1],
                           w[0]);
    return pdfY * pdfX;
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__ inline
void BlockPWLDistribution2D<TPB, X, Y>::DumpSharedMem(float* pdfX,
                                                      float* cdfX,
                                                      float* pdfY,
                                                      float* cdfY) const
{
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        uint32_t columnId = pixelId % X;

        // Only valid rows do the write
        if(rowId < Y)
        {
            pdfX[pixelId] = sMem.sPDFX[rowId][columnId];
            cdfX[pixelId] = sMem.sCDFX[rowId][columnId];
        }
    }
    // Dump Marginal PDF / CDF
    if(isColumnThread)
    {
        pdfY[threadId] = sMem.sPDFY[threadId];
        cdfY[threadId] = sMem.sCDFY[threadId];
    }
}