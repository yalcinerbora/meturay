#pragma once

#include "TracerCUDA/BlockSegmentedScan.cuh"
#include "cub/block/block_scan.cuh"

#include "RayLib/Vector.h"
#include "RNGenerator.h"

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
    __device__
    Vector2f    Sample(float& pdf, Vector2f& index, RNGeneratorGPUI& rng) const;
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
    __syncthreads();
    // All CDF's and PDFs are generated
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__ inline
Vector2f BlockPWCDistribution2D<TPB, X, Y>::Sample(float& pdf, Vector2f& index,
                                                   RNGeneratorGPUI& rng) const
{

}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__ inline
float BlockPWCDistribution2D<TPB, X, Y>::Pdf(const Vector2f& index) const
{

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