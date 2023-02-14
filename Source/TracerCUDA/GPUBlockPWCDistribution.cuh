#pragma once

#include "cub/block/block_scan.cuh"

#include "RayLib/Vector.h"
#include "RayLib/Constants.h"

#include "BlockSegmentedScan.cuh"
#include "RNGenerator.h"
#include "BinarySearch.cuh"

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
class BlockPWCDistribution2D
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
    static constexpr float DELTA_X          = 1.0f / X_FLOAT;
    static constexpr float DELTA_Y          = 1.0f / Y_FLOAT;

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

        // State is changing depending on the situation
        // PDF of CDF (implicit 1.0f) is omitted
        float sPCDFX[Y][X];
        float sPCDFY[Y];


//        float sCDFX[Y][X + 1];
//        float sPDFX[Y][X];
        // Y axis distribution data (PDF of CDF depending on the situation)
//        float sCDFY[Y + 1];
//        float sPDFY[Y];

    };

    private:
    TempStorage&    sMem;
    const uint32_t  threadId;
    const bool      isColumnThread;
    const bool      isMainThread;
    const bool      isRowLeader;
    const bool      isValidThread;
    // Whether shared memory holds the PDF or not
    bool            sharedHasPDF;

    // Each Thread holds either PDF of CDF
    // (whatever is not on the shared memory)
    float           pcdfRegisterX[DATA_PER_THREAD];
    float           pcdfRegisterY;

    protected:
    public:
    // Constructors & Destructor
    // Constructors PWC Distribution over the shared memory
    // Initially shared memory holds CDF
    __device__
                BlockPWCDistribution2D(TempStorage& storage,
                                       const float(&data)[DATA_PER_THREAD]);

    template <class RNG>
    __device__
    Vector2f    Sample(Vector2f& index, RNG& rng) const;

    __device__
    float       Pdf(const Vector2f& index) const;

    // Swap the register buffer to shared memory and vice versa
    __device__
    void        Swap();
    // Returns the shared memory situation
    __device__
    bool        IsPDFInShared() const;

    __device__
    void        DumpSharedMem(float* pdfXOut, float* cdfXOut,
                              float* pdfYOut, float* cdfYOut);
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
    , isValidThread(threadIdx.x < PIX_COUNT)
    , sharedHasPDF(false)
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
        // TODO: if entire warp is not contributing to the calculation
        // cull it entirely. Currently these warps just scan an array of zero
        // But also check if some synchronization may create a deadlock
        float cdfData;
        float totalSum;
        BlockSScan(sMem.algo.sSScanTempStorage).InclusiveSum(cdfData, totalSum,
                                                             pdfData, 0.0f);
        __syncthreads();
        // Row leader will do marginal PDF/CDF data
        // the Y Function value of this row
        if(isRowLeader && rowId < Y) sMem.sPCDFY[rowId] = totalSum;
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
        // Only valid rows do the write
        if(rowId < Y)
        {
            pcdfRegisterX[i] = pdfData;
            sMem.sPCDFX[rowId][columnId] = cdfData;
        }
    }
    __syncthreads();

    // Generate PDF CDF of the column (marginal pdf)
    //"sPDFY" variable holds the Y values
    float pdfDataY = (isColumnThread) ? sMem.sPCDFY[threadId]
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
    else if(isMainThread) printf("Entire 2D Histogram is empty!\n");

    // Expand the pdf back
    pdfDataY *= Y_FLOAT;

    if(isColumnThread)
    {
        sMem.sPCDFY[threadId] = cdfDataY;
        pcdfRegisterY = pdfDataY;
    }
    __syncthreads();
    // All CDF's and PDFs are generated
    // CDF's are on shared memory
    // PDF's are on registers
    // Need to swap to utilize PDF(...) function
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
template <class RNG>
__device__ inline
Vector2f BlockPWCDistribution2D<TPB, X, Y>::Sample<RNG>(Vector2f& index,
                                                        RNG& rng) const
{
    assert(!sharedHasPDF);

    Vector2f xi = rng.Uniform2D();
    // If entire distribution is invalid
    // Just sample uniformly
    if(sMem.sPCDFY[Y - 1] == 0.0f)
    {
        index = xi * Vector2f(X, Y);
        return xi;
    }
    bool foundY = GPUFunctions::BinarySearchInBetween<float>(index[1], xi[1],
                                                            sMem.sPCDFY, Y);

    // We are between 0 and first element, calculate accordingly
    if(!foundY && (xi[1] < sMem.sPCDFY[Y - 1]))
        index[1] = xi[1] / sMem.sPCDFY[0];
    // We exceed the CDF (probably due to numerical error)
    // clamp to edge
    else if(!foundY && (xi[1] >= sMem.sPCDFY[Y - 1]))
        index[1] = static_cast<float>(Y) - MathConstants::Epsilon;
    // We searched from first element so add one
    else
        index[1] += 1.0f;


    int32_t indexYInt = static_cast<int32_t>(index[1]);

    // Extremely rarely index becomes the light count
    // although Uniform should return [0, 1)
    // it still happens due to fp error i guess?
    // if it happens just return the last light on the list
    if(indexYInt >= Y)
    {
        printf("CUDA Error: Illegal Index on PwC Sample [Y = %.10f] [xi = %.10f], F: %s\n",
                         index[1], xi[1], foundY ? "True" : "False");
        indexYInt--;
    }

    const float* sRowCDF = sMem.sPCDFX[indexYInt];
    bool foundX = GPUFunctions::BinarySearchInBetween<float>(index[0], xi[0],
                                                             sRowCDF, X);
    // We are between 0 and first element, calculate accordingly
    if(!foundX && (xi[0] < sRowCDF[X - 1]))
        index[0] = xi[0] / sRowCDF[0];
    // We exceed the CDF (probably due to numerical error)
    // clamp to edge
    else if(!foundX && (xi[0] >= sRowCDF[X - 1]))
        index[0] = static_cast<float>(X) - MathConstants::Epsilon;
    // We searched from first element so add one
    else index[0] += 1.0f;

    int32_t indexXInt = static_cast<int32_t>(index[0]);
    if(indexXInt >= X)
    {
        printf("CUDA Error: Illegal Index on PwC Sample [X = %.10f] [xi = %.10f], F: %s\n",
               index[0], xi[0], foundX ? "True" : "False");
        indexXInt--;
    }
    // Return the index as a normalized coordinate as well
    return index * Vector2f(DELTA_X, DELTA_Y);
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__ inline
float BlockPWCDistribution2D<TPB, X, Y>::Pdf(const Vector2f& index) const
{
    assert(sharedHasPDF);
    Vector2ui indexInt = Vector2ui(static_cast<uint32_t>(index[0]),
                                   static_cast<uint32_t>(index[1]));
    float pdfX = sMem.sPCDFX[indexInt[1]][indexInt[0]];
    float pdfY = sMem.sPCDFY[indexInt[1]];

    if(pdfY == 0.0f || pdfX == 0)
    {
        printf("[Z] pdf(% .10f, % .10f), index(% .10f, % .10f) (% d, % d)\n",
               pdfX, pdfY, index[0], index[1], indexInt[0], indexInt[1]);
    }
    if(isnan(pdfX) || isnan(pdfY))
    {
        printf("[NaN] pdf(%.10f, %.10f), index (%.10f, %.10f) (%d, %d)\n",
               pdfX, pdfY, index[0], index[1], indexInt[0], indexInt[1]);
    }
    if(index.HasNaN())
    {
        printf("[NaN] index(%f, %f)\n", index[0], index[1]);
    }

    return pdfX * pdfY;
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__ inline
void BlockPWCDistribution2D<TPB, X, Y>::Swap()
{
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        //  Determine your row
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        uint32_t columnId = pixelId % X;

        if(pixelId < (X * Y))
        {
            float temp = sMem.sPCDFX[rowId][columnId];
            sMem.sPCDFX[rowId][columnId] = pcdfRegisterX[i];
            pcdfRegisterX[i] = temp;
        }
    }

    // Do the marginal
    if(isColumnThread)
    {
        float temp = sMem.sPCDFY[threadId];
        sMem.sPCDFY[threadId] = pcdfRegisterY;
        pcdfRegisterY = temp;
    }
    sharedHasPDF = !sharedHasPDF;
    __syncthreads();
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__ inline
bool BlockPWCDistribution2D<TPB, X, Y>::IsPDFInShared() const
{
    return sharedHasPDF;
}

template<uint32_t TPB,
         uint32_t X,
         uint32_t Y>
__device__ inline
void BlockPWCDistribution2D<TPB, X, Y>::DumpSharedMem(float* pdfX,
                                                      float* cdfX,
                                                      float* pdfY,
                                                      float* cdfY)
{
    assert(!sharedHasPDF);

    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        uint32_t columnId = pixelId % X;
        uint32_t cdfId = rowId * (X + 1) + (columnId + 1);

        // Only valid rows do the write
        if(rowId < Y)
        {

            cdfX[cdfId] = sMem.sPCDFX[rowId][columnId];
            // Don't forget to add the first data (which is zero)
            if(isRowLeader) cdfX[rowId * (X + 1)] = 0.0f;
        }
    }
    // Dump Marginal PDF / CDF
    if(isColumnThread)
    {
        cdfY[threadId + 1] = sMem.sPCDFY[threadId];
    }
    // Don't forget to write the first cdf y
    if(isMainThread) cdfY[0] = 0.0f;

    // Swap the buffers
    Swap();
    // Now do the PDF

    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        uint32_t columnId = pixelId % X;
        if(rowId < Y)
        {
            pdfX[pixelId] = sMem.sPCDFX[rowId][columnId];
        }

    }
    // Dump Marginal PDF / CDF
    if(isColumnThread)
    {
        pdfY[threadId] = sMem.sPCDFY[threadId];
    }
}