#pragma once

#include "cub/block/block_scan.cuh"

#include "RayLib/Vector.h"

#include "BlockSegmentedScan.cuh"
#include "RNGenerator.h"
#include "BinarySearch.cuh"

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
class BlockPWCDistribution2D
{
    private:
    // No SFINAE, just static assert
    static_assert(TPB <= (X * Y), "TPB should be less than or equal of (X * Y)");
    static_assert((X * Y) % TPB  == 0, "(X * Y) must be multiple of TPB");

    static constexpr uint32_t DATA_PER_THREAD = (X * Y) / TPB;


    static constexpr float X_FLOAT = static_cast<float>(X);
    static constexpr float Y_FLOAT = static_cast<float>(Y);
    static constexpr float DELTA_X = 1.0f / X_FLOAT;
    static constexpr float DELTA_Y = 1.0f / Y_FLOAT;

    using BlockSScan    = BlockSegmentedScan<float, TPB, X>;
    using BlockScan     = cub::BlockScan<float, Y>;

    public:
    struct TempStorage
    {
        union
        {
            typename BlockSScan::TempStorage    sSScanTempStorage;
            typename BlockScan::TempStorage     sScanTempStorage;
        } algo;
        float sCDFX[Y][X + 1];
        float sPDFX[Y][X];
        // Y axis distribution data (PDF of CDF depending on the situation)
        float sCDFY[Y + 1];
        float sPDFY[Y];
    };

    private:
    TempStorage&    sMem;
    const uint32_t  threadId;
    const bool      isColumnThread;
    const bool      isMainThread;
    const bool      isRowLeader;

    protected:
    public:
    // Constructors & Destructor
    __device__
                BlockPWCDistribution2D(TempStorage& storage,
                                       const float(&data)[DATA_PER_THREAD]);

    template <class RNG>
    __device__
    Vector2f    Sample(float& pdf, Vector2f& index, RNG& rng) const;

    __device__
    float       Pdf(const Vector2f& index) const;

    __device__
    void        DumpSharedMem(float* pdfXOut, float* cdfXOut,
                              float* pdfYOut, float* cdfYOut) const;
};

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__
BlockPWCDistribution2D<TPB, X, Y>::BlockPWCDistribution2D(TempStorage& storage,
                                                          const float(&data)[DATA_PER_THREAD])
    : sMem(storage)
    , threadId(threadIdx.x)
    , isColumnThread(threadIdx.x < Y)
    , isMainThread(threadIdx.x == 0)
    , isRowLeader((threadIdx.x % X) == 0)
{
    // Initialize the CDF/PDF
    // Generate PWC Distribution over the date
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        float pdfData = data[i];
        //  Each thread in block will contribute a different row
        //  Determine your row
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        uint32_t columnId = pixelId % X;

        // Scan operation to generate CDF
        float cdfData;
        float totalSum;
        BlockSScan(sMem.algo.sSScanTempStorage).InclusiveSum(cdfData, totalSum,
                                                             pdfData, 0.0f);
        __syncthreads();
        // Row leader will do marginal PDF/CDF data
        // the Y Function value of this row
        if(isRowLeader) sMem.sPDFY[rowId] = totalSum;
        // Now normalize the pdf/cdf with the dimension
        // Getting ready for the scan operation
        cdfData *= DELTA_X;
        totalSum *= DELTA_X;
        // Do the normalization for PDF and CDF
        if(totalSum != 0.0f)
        {
            // If total sum is zero
            // meaning that this row is not probable
            // Prevent NaN here
            pdfData *= (1.0f / totalSum);
            cdfData *= (1.0f / totalSum);
        }
        sMem.sPDFX[rowId][columnId] = pdfData;
        sMem.sCDFX[rowId][columnId + 1] = cdfData;
        if(isRowLeader) sMem.sCDFX[rowId][0] = 0.0f;
    }
    __syncthreads();

    // Generate PDF CDF of the column (marginal pdf)
    //"sPDFY" variable holds the Y values
    float pdfDataY = (isColumnThread) ? sMem.sPDFY[threadId]
                                      : 0.0f;
    // Now normalize the pdf with the dimension
    // Getting ready for the scan operation
    pdfDataY *= DELTA_Y;
    // Scan operation to generate CDF
    float cdfDataY;
    float totalSum;
    BlockScan(sMem.algo.sScanTempStorage).InclusiveSum(pdfDataY, cdfDataY, totalSum);
    __syncthreads();

    // Do the normalization for PDF and CDF
    if(totalSum != 0.0f)
    {
        // If total sum is zero
        // meaning that this row is not probable
        // Prevent NaN here
        pdfDataY *= (1.0f / totalSum);
        cdfDataY *= (1.0f / totalSum);
    }
    // Expand the pdf back
    pdfDataY *= Y_FLOAT;

    if(isColumnThread)
    {
        sMem.sCDFY[threadId + 1] = cdfDataY;
        sMem.sPDFY[threadId] = pdfDataY;
        if(isMainThread) sMem.sCDFY[0] = 0.0f;
    }
    //if(isMainThread)
    //{
    //    printf("CDF\n"
    //           "%f, %f, %f, %f, %f, %f, %f, %f\n"
    //           "%f, %f, %f, %f, %f, %f, %f, %f\n"
    //           "%f, %f, %f, %f, %f, %f, %f, %f\n"
    //           "%f, %f, %f, %f, %f, %f, %f, %f\n"
    //           "%f\n",
    //           sMem.sCDFY[0], sMem.sCDFY[1], sMem.sCDFY[2], sMem.sCDFY[3],
    //           sMem.sCDFY[4], sMem.sCDFY[5], sMem.sCDFY[6], sMem.sCDFY[7],
    //           sMem.sCDFY[8], sMem.sCDFY[9], sMem.sCDFY[10], sMem.sCDFY[11],
    //           sMem.sCDFY[12], sMem.sCDFY[13], sMem.sCDFY[14], sMem.sCDFY[15],
    //           sMem.sCDFY[16], sMem.sCDFY[17], sMem.sCDFY[18], sMem.sCDFY[19],
    //           sMem.sCDFY[20], sMem.sCDFY[21], sMem.sCDFY[22], sMem.sCDFY[23],
    //           sMem.sCDFY[24], sMem.sCDFY[25], sMem.sCDFY[26], sMem.sCDFY[27],
    //           sMem.sCDFY[28], sMem.sCDFY[29], sMem.sCDFY[30], sMem.sCDFY[31],
    //           sMem.sCDFY[32]);
    //}

    __syncthreads();
    // All CDF's and PDFs are generated
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
template <class RNG>
__device__ inline
Vector2f BlockPWCDistribution2D<TPB, X, Y>::Sample<RNG>(float& pdf, Vector2f& index,
                                                        RNG& rng) const
{
    static constexpr int32_t CDF_SIZE_Y = Y + 1;
    static constexpr int32_t CDF_SIZE_X = X + 1;

    Vector2f xi = rng.Uniform2D();
    // If entire distribution is invalid
    // Just sample uniformly
    if(sMem.sCDFY[Y] == 0.0f)
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
    //if(indexYInt == CDF_SIZE_Y)
    if(indexYInt == CDF_SIZE_Y)
    {
        KERNEL_DEBUG_LOG("CUDA Error: Illegal Index on PwC Sample [Y = %f]\n",
                         index[1]);
        indexYInt--;
    }
    const float* sRowCDF = sMem.sCDFX[indexYInt];
    const float* sRowPDF = sMem.sPDFX[indexYInt];

    GPUFunctions::BinarySearchInBetween<float>(index[0], xi[0],
                                               sRowCDF, CDF_SIZE_X);
    int32_t indexXInt = static_cast<int32_t>(index[0]);

    // Samples are dependent so we need to multiply the pdf results
    pdf = sMem.sPDFY[indexYInt] * sRowPDF[indexXInt];
    // Return the index as a normalized coordinate as well
    return index * Vector2f(DELTA_X, DELTA_Y);
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__ inline
float BlockPWCDistribution2D<TPB, X, Y>::Pdf(const Vector2f& index) const
{
    Vector2ui indexInt = Vector2ui(static_cast<uint32_t>(index[0]),
                                   static_cast<uint32_t>(index[1]));

    return sMem.sPDFY[indexInt[1]] * sMem.sPDFX[indexInt[1]][indexInt[0]];
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__ inline
void BlockPWCDistribution2D<TPB, X, Y>::DumpSharedMem(float* pdfX,
                                                      float* cdfX,
                                                      float* pdfY,
                                                      float* cdfY) const
{
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        uint32_t columnId = pixelId % X;
        uint32_t cdfId = rowId * (X + 1) + (columnId + 1);

        pdfX[pixelId] = sMem.sPDFX[rowId][columnId];
        cdfX[cdfId] = sMem.sCDFX[rowId][columnId + 1];
        // Don't forget to add the first data (which should be zero)
        if(isRowLeader) cdfX[rowId * (X + 1)] = sMem.sCDFX[rowId][0];
    }
    // Dump Marginal PDF / CDF
    if(isColumnThread)
    {
        pdfY[threadId] = sMem.sPDFY[threadId];
        cdfY[threadId + 1] = sMem.sCDFY[threadId + 1];
    }
    // Don't forget to write the first cdf y
    if(isMainThread) cdfY[0] = sMem.sCDFY[0];
}