#pragma once

#include "RayLib/Types.h"
#include "RayLib/ColorConversion.h"

#include "GPUMetaSurfaceGenerator.h"
#include "GPUMetaSurface.h"
#include "GPUMediumVacuum.cuh"
#include "TracerConstants.h"

#include <cub/cub.cuh>

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
class ProductSampler
{
    public:
    static constexpr Vector2i PRODUCT_MAP_SIZE = Vector2i(PX, PY);
    static constexpr int32_t WARP_PER_BLOCK = TPB / WARP_SIZE;
    static constexpr int32_t REFL_PER_THREAD = (PX * PY) / WARP_SIZE;
    static constexpr int32_t PROCESSED_ROW_PER_ITER = WARP_SIZE / PX;
    static_assert(TPB % WARP_SIZE == 0);
    static_assert(REFL_PER_THREAD / WARP_SIZE == 0, "Reflectance field should be divisible by warp size(32)");
    // Radiance Field related
    static constexpr int32_t RADIANCE_PER_THREAD = std::max(1, (X * Y) / TPB);
    static constexpr int32_t SMALL_RADIANCE_PER_THREAD = std::max(1, (PX * PY) / TPB);
    // Inner field related
    static constexpr Vector2i DATA_PER_PRODUCT = Vector2i(X / PX, Y / PY);
    static constexpr int32_t DATA_PER_PRODUCT_LINEAR = DATA_PER_PRODUCT[0] * DATA_PER_PRODUCT[1];
    static constexpr int32_t WARP_ITERATION_COUNT = std::max(1, (DATA_PER_PRODUCT_LINEAR) / WARP_SIZE);

    static constexpr int32_t DetermineLogicalWarpForReduce()
    {
        // CUB does not like single logical warp for reduce operation
        // which does not make sense anyway
        return std::max(2, std::min(DATA_PER_PRODUCT_LINEAR, WARP_SIZE));
    }

    static constexpr float PX_FLOAT = static_cast<float>(PX);
    static constexpr float PY_FLOAT = static_cast<float>(PY);

    using WarpReduce = cub::WarpReduce<float, DetermineLogicalWarpForReduce()>;
    using WarpRowScan = cub::WarpScan<float, PX>;
    using WarpColumnScan = cub::WarpScan<float, PY>;
    struct SharedStorage
    {
        union
        {
            typename WarpReduce::TempStorage reduceMem[WARP_PER_BLOCK];
            typename WarpRowScan::TempStorage rowScanMem[WARP_PER_BLOCK];
            typename WarpColumnScan::TempStorage colScanMem[WARP_PER_BLOCK];
        };

        // Main radiance field
        float sRadianceField[X][Y];
        // Normalization constant (for the inner field region)
        // Rejection sampling will be use this (this is the maximum value in the region)
        float sNormalizationConstants[PRODUCT_MAP_SIZE[0]][PRODUCT_MAP_SIZE[1]];
        // Reduced radiance field (will be multiplied by the BxDF field)
        float sRadianceFieldSmall[PRODUCT_MAP_SIZE[0]][PRODUCT_MAP_SIZE[1]];
        // Scratch data (hold pdfY temporarily on sampling (product field's)
        // also holds xi0, xi1 (random numbers) selected cdf ranges, and pdfX and pdfY
        // for the product region
        float sWarpTempMemory[WARP_PER_BLOCK][std::max(8, PY)];
        // Meta Surface for each warp
        Byte sSurfacesRaw[WARP_PER_BLOCK * sizeof(GPUMetaSurface)];
    };

    private:
    SharedStorage&                          shMem;

    const RayId*                            gRayIds;
    const GPUMetaSurfaceGeneratorGroup&     metaSurfGenerator;
    uint32_t                                rayCount;

    // Parallelization Logic Related
    int32_t                                 threadId;
    int32_t                                 warpLocalId;
    int32_t                                 warpId;
    bool                                    isWarpLeader;
    bool                                    isRowLeader;

    public:
    __device__      ProductSampler(SharedStorage& sharedMem,
                                   const float(&radiances)[RADIANCE_PER_THREAD],
                                   const RayId* gRayIds,
                                   const GPUMetaSurfaceGeneratorGroup& generator);

    template <class ProjFunc>
    __device__
    Vector2f        SampleProduct(float& pdf,
                                  RNGeneratorGPUI& rng,
                                  uint32_t rayIndex,
                                  ProjFunc&& Project) const;


    // For Test and Debugging
    __device__
    void            DumpRadianceField(float* radianceField);
    __device__
    void            DumpSmallRadianceField(float* smallRadianceField);
    __device__
    void            DumpNormalizationConstants(float* normConstants);

};

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
__device__ __forceinline__
ProductSampler<TPB, X, Y, PX, PY>::ProductSampler(// Temp
                                                  SharedStorage& sharedMem,
                                                  // Inputs
                                                  const float(&radiances)[RADIANCE_PER_THREAD],
                                                  const RayId* gRayIds,
                                                  const GPUMetaSurfaceGeneratorGroup& generator)
    : shMem(sharedMem)
    , gRayIds(gRayIds)
    , metaSurfGenerator(generator)
    , threadId(threadIdx.x)
    , warpLocalId(threadIdx.x % WARP_SIZE)
    , warpId(threadIdx.x / WARP_SIZE)
    , isWarpLeader(warpLocalId == 0)
    , isRowLeader((warpLocalId% PX) == 0)
{
    for(int32_t i = 0; i < RADIANCE_PER_THREAD; i++)
    {
        // Find the row column
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        uint32_t columnId = pixelId % X;

        shMem.sRadianceField[rowId][columnId] = radiances[i];
    };
    __syncthreads();

    // Change the logic, a warp will be responsible for reducing
    // (X / PX) * (Y / PY) to a single cell
    // (i.e. 64x64, 8x8 product => 64 adjacent threads)
    // will reduce to a single value
    for(int32_t j = warpId; j < PRODUCT_MAP_SIZE.Multiply(); j += WARP_PER_BLOCK)
    {
        // Find the cell from the warp id
        uint32_t linearCellId = j;
        Vector2i cellId2D = Vector2i(linearCellId % PX,
                                     linearCellId / PX);

        // Warp reduce N times
        float maxValue = 0.0f;
        float totalReduce = 0.0f;
        for(int i = 0; i < WARP_ITERATION_COUNT; i++)
        {
            int32_t linearInnerId = warpLocalId + i * WARP_SIZE;
            Vector2i innerId2D = Vector2i(linearInnerId % DATA_PER_PRODUCT[0],
                                          linearInnerId / DATA_PER_PRODUCT[0]);

            Vector2i combinedId2D = cellId2D + innerId2D;

            float value = (linearInnerId < DATA_PER_PRODUCT_LINEAR)
                                ? shMem.sRadianceField[combinedId2D[1]][combinedId2D[0]]
                                : 0.0f;


            totalReduce += WarpReduce(shMem.reduceMem[warpId]).Sum(value);
            float newMax = WarpReduce(shMem.reduceMem[warpId]).Reduce(value, cub::Max());
            maxValue = max(maxValue, newMax);
        }
        // Store the reduced value (averaged)
        if(isWarpLeader && linearCellId < (PX * PY))
        {
            float dataPerProduct = static_cast<float>(DATA_PER_PRODUCT_LINEAR);
            shMem.sRadianceFieldSmall[cellId2D[1]][cellId2D[0]] = totalReduce / dataPerProduct;
            shMem.sNormalizationConstants[cellId2D[1]][cellId2D[0]] = maxValue;
        }
    }
    __syncthreads();
}

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
template <class ProjFunc>
__device__ __forceinline__
Vector2f ProductSampler<TPB, X, Y, PX, PY>::SampleProduct(float& pdf,
                                                          RNGeneratorGPUI& rng,
                                                          uint32_t rayIndex,
                                                          ProjFunc&& Project) const
{
    // Convert Raw MetaSurface memory
    // TODO: bad design change this later
    //GPUMetaSurface* sSurfaces = reinterpret_cast<GPUMetaSurface*>(shMem.sSurfacesRaw);

    //Vector3f wo = YAxis;
    // Load material & surface to shared mem
    // Call copy construct here (GPUMetaSurface has reference)
    //if(isWarpLeader)
    //{
    //    // Meta Surface is large to hold in register space use shared memory instead
    //    new (sSurfaces + warpId) GPUMetaSurface(metaSurfGenerator.AcquireWork(gRayIds[rayIndex]));
    //    wo = -(metaSurfGenerator.Ray(gRayIds[rayIndex]).ray.getDirection());
    //}
    // Broadcast the outgoing direction to peers
    //wo[0] = __shfl_sync(0xFFFFFFFF, wo[0], 0, WARP_SIZE);
    //wo[1] = __shfl_sync(0xFFFFFFFF, wo[1], 0, WARP_SIZE);
    //wo[2] = __shfl_sync(0xFFFFFFFF, wo[2], 0, WARP_SIZE);

    // No need to sync here, threads are in lockstep in warp
    //const GPUMetaSurface& surface = sSurfaces[warpId];
    // TODO: Change this to a specific medium, current is does not work
    //const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);
    GPUMediumVacuum medium(0);

    // Generate PX by PY product field (store it on register space)
    float productField[REFL_PER_THREAD];
    //for(int32_t i = warpId; i < (PX * PY); i += WARP_SIZE)
    for(int32_t j = 0; j < REFL_PER_THREAD; j++)
    {
        int32_t linearId = warpLocalId + (j * PROCESSED_ROW_PER_ITER);

        Vector2i loc2D = Vector2i(linearId % PX,
                                  linearId / PX);

        // Only calculate if surface is not specular
        Vector3f bxdfColored = Vector3f(1.0f);
        //if(surface.Specularity() < TracerConstants::SPECULAR_THRESHOLD)
        {
            // Project the mapped id
            //Vector3f wi = Project(loc2D, Vector2i(PX, PY));
            //bxdfColored = surface.Evaluate(wo,
            //                               wi,
            //                               medium);
            bxdfColored = Vector3f(static_cast<float>(warpId * WARP_SIZE + j));
        }

        productField[j] = (1.0f * //Utility::RGBToLuminance(bxdfColored) *
                           shMem.sRadianceFieldSmall[loc2D[1]][loc2D[0]]);
    }

    // Our field is generated now sample,
    // For first stage do a CDF parallel search search style
    // approach
    float cdfX[REFL_PER_THREAD];
    float pdfX[REFL_PER_THREAD];
    for(int32_t j = 0; j < REFL_PER_THREAD; j++)
    {
        float rowTotal;
        WarpRowScan(shMem.rowScanMem[warpId]).InclusiveSum(productField[j],
                                                           cdfX[j], rowTotal);

        float rowTotalRecip = 1.0f / rowTotal;
        cdfX[j] *= rowTotalRecip;
        pdfX[j] = productField[j] * PX_FLOAT * rowTotalRecip;
        // CDF per lane is done
        // Store the total value for marginal pdf
        // rowLeader writes the data
        int32_t rowIndex = (warpLocalId / PX) + (j * PROCESSED_ROW_PER_ITER);
        if(warpLocalId % PX == 0) shMem.sWarpTempMemory[warpId][rowIndex] = rowTotal;
    }

    // Now calculate Marginal
    // Assuming PY fits in to a warp (it should be)
    float colValue = (warpLocalId < PY) ? shMem.sWarpTempMemory[warpId][warpLocalId] : 0.0f;
    float columnTotal, columnScan;
    WarpColumnScan(shMem.colScanMem[warpId]).InclusiveSum(colValue, columnScan,
                                                          columnTotal);
    float columnTotalRecip = 1.0f / columnTotal;

    float cdfY = columnScan * columnTotalRecip;
    float pdfY = (warpLocalId < PY) ? shMem.sWarpTempMemory[warpId][warpLocalId] : 0.0f;
    pdfY *= PY_FLOAT * columnTotalRecip;

    // Generated Marginal & Conditional Probabilities
    // Parallel search the CDFY
    float xi0 = (isWarpLeader) ? rng.Uniform() : 0.0f;
    xi0 = __shfl_sync(0xFFFFFFFF, xi0, 0, PY);
    // Check the votes
    uint32_t mask = __ballot_sync(0xFFFFFFFF, (xi0 > cdfY));
    int32_t rowId = __ffs(~mask) - 1;

    // save the xi_Y, (cdfY_0, cdfY_1]
    //                           ^  rowId is the index of this one
    if(isWarpLeader)
        shMem.sWarpTempMemory[warpId][0] = xi0;
    if(warpLocalId == rowId)
    {
        shMem.sWarpTempMemory[warpId][2] = cdfY;
        shMem.sWarpTempMemory[warpId][3] = pdfY;
    }
    if(rowId && warpLocalId == (rowId - 1))
        shMem.sWarpTempMemory[warpId][1] = cdfY;
    if((!rowId) && isWarpLeader)
        shMem.sWarpTempMemory[warpId][1] = 0.0f;

    // Now do the same but for row
    int32_t myColumnId = (warpLocalId % PX);
    int32_t myRowIndex = (warpLocalId / PX);
    int32_t reducedRow = rowId % PROCESSED_ROW_PER_ITER;
    int accessIndex = rowId / PROCESSED_ROW_PER_ITER;
    bool isMyRow = (myRowIndex == reducedRow);

    // Broadcast the random number (for dimension X)
    float xi1 = (isRowLeader && isMyRow) ? rng.Uniform() : 0.0f;
    xi1 = __shfl_sync(0xFFFFFFFF, xi1, 0, PX);
    // Check votes
    mask = __ballot_sync(0xFFFFFFFF, (xi1 > cdfX[accessIndex]));
    int columnId = __ffs(~mask) - 1;
    // save the xi_X, (cdfX_0, cdfX_1]
    //                           ^  column is the index of this one
    if(isMyRow)
    {
        if(isRowLeader)
            shMem.sWarpTempMemory[warpId][4] = xi1;
        if(myColumnId == columnId)
        {
            shMem.sWarpTempMemory[warpId][6] = cdfX[accessIndex];
            shMem.sWarpTempMemory[warpId][7] = pdfX[accessIndex];
        }
        if(columnId && myColumnId == (columnId - 1))
            shMem.sWarpTempMemory[warpId][5] = cdfX[accessIndex];
        if((!columnId) && isRowLeader)
            shMem.sWarpTempMemory[warpId][5] = 0.0f;
    }
    // Product sampling phase is done!
    // now subsample the inner small region (could be same size)

    // Here we will use rejection sampling, it is highly parallel,
    // each warp will do one round of roll check if we could not find
    // anything warp leader will sample uniformly
    // Rejection sampling
    // https://www.realtimerendering.com/raytracinggems/rtg/index.html Chapter 16
    Vector2f uv = rng.Uniform2D();
    float xiTest = rng.Uniform();

    static constexpr Vector2f DPP_FLOAT(DATA_PER_PRODUCT[0], DATA_PER_PRODUCT[1]);
    Vector2i outerRegion2D = Vector2i(columnId, rowId);
    Vector2i innerRegion2D = Vector2i(uv * DPP_FLOAT);
    Vector2i index2D = outerRegion2D + innerRegion2D;

    float texVal = shMem.sRadianceField[index2D[1]][index2D[0]];
    texVal /= shMem.sNormalizationConstants[outerRegion2D[1]][outerRegion2D[0]];

    mask = __ballot_sync(0xFFFFFFFF, (xiTest > texVal));
    int32_t luckyWarp = __ffs(mask) - 1;

    // Broadcast to the leader warp (well to everybody as well due to
    // instruction)
    if(luckyWarp >= 0)
    {
        uv[0] = __shfl_sync(0xFFFFFFFF, uv[0], luckyWarp, WARP_SIZE);
        uv[1] = __shfl_sync(0xFFFFFFFF, uv[1], luckyWarp, WARP_SIZE);
        texVal = __shfl_sync(0xFFFFFFFF, texVal, luckyWarp, WARP_SIZE);
    }
    if(isWarpLeader)
    {
        float integral = shMem.sRadianceFieldSmall[outerRegion2D[1]][outerRegion2D[0]];
        // Now calculate actual pdf etc.
        float pdfInner = (luckyWarp >= 0) ? texVal / integral
                                          : 1.0f;
        // Finally actual pdf
        pdf = shMem.sWarpTempMemory[warpId][7] *
              shMem.sWarpTempMemory[warpId][3] * pdfInner;
    }
    // Calculate the UV
    Vector2f outerUV = Vector2f(outerRegion2D);
    outerUV += uv;
    // All done!
    return outerUV;
}

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
__device__ inline
void ProductSampler<TPB, X, Y, PX, PY>::DumpRadianceField(float* radianceField)
{
    for(int32_t i = 0; i < RADIANCE_PER_THREAD; i++)
    {
        // Find the row column
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        uint32_t columnId = pixelId % X;

        if(pixelId < X * Y)
            radianceField[pixelId] = shMem.sRadianceField[rowId][columnId];
    }
}

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
__device__ inline
void ProductSampler<TPB, X, Y, PX, PY>::DumpSmallRadianceField(float* smallRadianceField)
{
    for(int32_t i = 0; i < SMALL_RADIANCE_PER_THREAD; i++)
    {
        // Find the row column
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / PX;
        uint32_t columnId = pixelId % PX;

        if(pixelId < PX * PY)
            smallRadianceField[pixelId] = shMem.sRadianceFieldSmall[rowId][columnId];
    }
}

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
__device__ inline
void ProductSampler<TPB, X, Y, PX, PY>::DumpNormalizationConstants(float* normConstants)
{
    for(int32_t i = 0; i < SMALL_RADIANCE_PER_THREAD; i++)
    {
        // Find the row column
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / PX;
        uint32_t columnId = pixelId % PX;

        if(pixelId < PX * PY)
            normConstants[pixelId] = shMem.sNormalizationConstants[rowId][columnId];
    }
}