#pragma once

#include "RayLib/Types.h"
#include "RayLib/ColorConversion.h"

#include "GPUMetaSurfaceGenerator.h"
#include "GPUMetaSurface.h"
#include "GPUMediumVacuum.cuh"
#include "TracerConstants.h"
#include "TracerFunctions.cuh"

#include <cub/cub.cuh>

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
class ProductSampler
{
    public:
    static constexpr bool ProductFieldFitCheck()
    {
        auto PIX_COUNT = (PX * PY);
        if(WARP_SIZE > PIX_COUNT) return WARP_SIZE % PIX_COUNT == 0;
        if(WARP_SIZE <= PIX_COUNT) return PIX_COUNT % WARP_SIZE == 0;
        return false;
    }

    static constexpr Vector2i PRODUCT_MAP_SIZE = Vector2i(PX, PY);
    static constexpr int32_t WARP_PER_BLOCK = TPB / WARP_SIZE;
    static constexpr int32_t REFL_PER_THREAD = std::max(1, (PX * PY) / WARP_SIZE);
    static constexpr int32_t PROCESSED_ROW_PER_ITER = WARP_SIZE / PX;
    // Radiance Field related
    static constexpr int32_t RADIANCE_PER_THREAD = std::max(1, (X * Y) / TPB);
    static constexpr int32_t SMALL_RADIANCE_PER_THREAD = std::max(1, (PX * PY) / TPB);
    // Inner field related
    static constexpr Vector2i DATA_PER_PRODUCT = Vector2i(X / PX, Y / PY);
    static constexpr int32_t DATA_PER_PRODUCT_LINEAR = DATA_PER_PRODUCT[0] * DATA_PER_PRODUCT[1];
    static constexpr int32_t WARP_ITERATION_COUNT = std::max(1, (DATA_PER_PRODUCT_LINEAR) / WARP_SIZE);
    // CUB does not like single logical warp for reduce operation,
    // which does not make sense anyway so we will fake it using two logical warps and
    // one having the identity element (zero) for reduction operation (add)
    // TODO: this is changed in latest cub, change this later
    static constexpr int32_t LOGICAL_WARP_COUNT_FOR_REDUCE = std::max(2, std::min(DATA_PER_PRODUCT_LINEAR,
                                                                                  WARP_SIZE));
    static_assert((PX <= WARP_SIZE) && (PY <= WARP_SIZE),
                  "product field's single row or column should fit inside a warp");
    static_assert(TPB % WARP_SIZE == 0, "TPB should be a multiple of warp size(32)");
    static_assert(ProductFieldFitCheck(),
                  "Reflectance field should be evenly divisible by warp size(32)");
    static_assert(X >= PX && Y >= PY, "Product field cannot be larger than the actual field.");

    static constexpr float PX_FLOAT = static_cast<float>(PX);
    static constexpr float PY_FLOAT = static_cast<float>(PY);

    using WarpReduce = cub::WarpReduce<float, LOGICAL_WARP_COUNT_FOR_REDUCE>;
    using WarpRowScan = cub::WarpScan<float, PX>;
    using WarpColumnScan = cub::WarpScan<float, PY>;
    struct SharedStorage
    {
        // Divide the warp local memory as AoS pattern
        // since each warp may use different fundamental operation
        // at the same time
        union
        {
            typename WarpReduce::TempStorage        reduceMem;
            typename WarpRowScan::TempStorage       rowScanMem;
            typename WarpColumnScan::TempStorage    colScanMem;
        } cubMem[WARP_PER_BLOCK];

        // Main radiance field
        float sRadianceField[X][Y];
        // Normalization constant (for the inner field region)
        // Rejection sampling will use this (this is the maximum value in the region)
        float sNormalizationConstants[PRODUCT_MAP_SIZE[0]][PRODUCT_MAP_SIZE[1]];
        // Reduced radiance field (will be multiplied by the BxDF field)
        float sRadianceFieldSmall[PRODUCT_MAP_SIZE[0]][PRODUCT_MAP_SIZE[1]];
        // Scratch data (hold pdfY temporarily on sampling (product field's)
        // also holds xi0, xi1 (random numbers) selected cdf ranges, and pdfX and pdfY
        // for the product region
        float sWarpTempMemory[WARP_PER_BLOCK][std::max(2, PY)];
        // Meta Surface for each warp
        GPUMetaSurface sSurfaces[WARP_PER_BLOCK];
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


    template <class ProjFunc>
    __device__ void                         MultiplyWithBXDF(float(&productField)[REFL_PER_THREAD],
                                                             int32_t rayIndex, ProjFunc&&) const;

    template <class ProjFunc>
    __device__ void                         GeneratePDFAndCDF(float(&pdfX)[REFL_PER_THREAD],
                                                              float(&cdfX)[REFL_PER_THREAD],
                                                              float& pdfY,
                                                              float& cdfY,
                                                              int32_t rayIndex,
                                                              ProjFunc&& Project) const;

    __device__ float                        Pdf(const Vector2f& uv,
                                                const float(&pdfX)[REFL_PER_THREAD],
                                                const float pdfY,
                                                bool innerWasUniform) const;

    __device__ Vector2f                     Sample(bool& innerIsUniform, float& pdfOut,
                                                   RNGeneratorGPUI& rng,
                                                   const float(&pdfX)[REFL_PER_THREAD],
                                                   const float(&cdfX)[REFL_PER_THREAD],
                                                   const float& pdfY,
                                                   const float& cdfY) const;

    public:
    // Constructors & Destructor
    __device__      ProductSampler(SharedStorage& sharedMem,
                                   const float(&radiances)[RADIANCE_PER_THREAD],
                                   const RayId* gRayIds,
                                   const GPUMetaSurfaceGeneratorGroup& generator);

    // Sample Function
    template <class ProjFunc>
    __device__
    Vector2f        SampleWithProduct(float& pdf,
                                      RNGeneratorGPUI& rng,
                                      int32_t rayIndex,
                                      ProjFunc&& Project) const;
    template <class ProjFunc, class InvProjFunc, class NormProjFunc>
    __device__
    Vector2f        SampleMIS(float& pdf,
                              RNGeneratorGPUI& rng,
                              int32_t rayIndex,
                              ProjFunc&& Project,
                              InvProjFunc&& InvProject,
                              NormProjFunc&& NormProject,
                              float sampleRatio,
                              float projectionPdfMultiplier) const;


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
    , isRowLeader((warpLocalId % PX) == 0)
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

            static constexpr Vector2i DPP = DATA_PER_PRODUCT;
            Vector2i combinedId2D = cellId2D * DPP + innerId2D;

            float value = (linearInnerId < DATA_PER_PRODUCT_LINEAR)
                                ? shMem.sRadianceField[combinedId2D[1]][combinedId2D[0]]
                                : 0.0f;

            totalReduce += WarpReduce(shMem.cubMem[warpId].reduceMem).Sum(value);
            float newMax = WarpReduce(shMem.cubMem[warpId].reduceMem).Reduce(value, cub::Max());
            maxValue = max(maxValue, newMax);
        }
        // Store the reduced value (averaged)
        if(isWarpLeader && linearCellId < (PX * PY))
        {
            float dataPerProduct = static_cast<float>(DATA_PER_PRODUCT_LINEAR);
            shMem.sRadianceFieldSmall[cellId2D[1]][cellId2D[0]] = totalReduce / dataPerProduct;
            shMem.sNormalizationConstants[cellId2D[1]][cellId2D[0]] = maxValue;

            if(totalReduce / dataPerProduct == 0.0f) printf("ProductRegion is zero!\n");
        }
    }
    __syncthreads();
}

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
template <class ProjFunc>
__device__ __forceinline__
void ProductSampler<TPB, X, Y, PX, PY>::MultiplyWithBXDF(float(&productField)[REFL_PER_THREAD],
                                                         int32_t rayIndex, ProjFunc&& Project) const
{
    static constexpr int32_t INVALID_RAY_INDEX = -1;
    Vector3f wi = YAxis;
    // Load material & surface to shared mem
    // Call copy construct here (GPUMetaSurface has reference)
    if(rayIndex != INVALID_RAY_INDEX)
    {
        if(isWarpLeader)
        {
            // Meta Surface is too large to hold in register space use shared memory instead
            shMem.sSurfaces[warpId] = metaSurfGenerator.AcquireWork(gRayIds[rayIndex]);
            wi = -(metaSurfGenerator.Ray(gRayIds[rayIndex]).ray.getDirection());

            //if(rayIndex == 0)
            //    printf("Loaded Surface L(%s): WN:(%f, %f, %f)\n",
            //           shMem.sSurfaces[warpId].IsLight() ? "true" : "false",
            //           shMem.sSurfaces[warpId].WorldNormal()[0],
            //           shMem.sSurfaces[warpId].WorldNormal()[1],
            //           shMem.sSurfaces[warpId].WorldNormal()[2]);
        }
        // Broadcast the outgoing direction to peers
        wi[0] = __shfl_sync(0xFFFFFFFF, wi[0], 0, WARP_SIZE);
        wi[1] = __shfl_sync(0xFFFFFFFF, wi[1], 0, WARP_SIZE);
        wi[2] = __shfl_sync(0xFFFFFFFF, wi[2], 0, WARP_SIZE);
    }

    const GPUMetaSurface& warpSurf = shMem.sSurfaces[warpId];

    // Generate PX by PY product field (store it on register space)
    for(int32_t j = 0; j < REFL_PER_THREAD; j++)
    {
        int32_t linearId = warpLocalId + (j * WARP_SIZE);
        // Potential case when product field is small 4x4 for example.
        Vector2i loc2D = Vector2i(linearId % PX, linearId / PX);
        float bxdfGray = 1.0f;
        // Only calculate if surface is not specular
        // or we are skipping the product portion
        if(rayIndex != INVALID_RAY_INDEX &&
           (!warpSurf.IsLight()) &&
           warpSurf.Specularity() < TracerConstants::SPECULAR_THRESHOLD)
        {
            // TODO: Change this to a specific medium, currently this does not work
            // if there are medium changes.
            //const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);
            GPUMediumVacuum medium(0);
            // Project the mapped id to world space
            Vector3f wo = Project(loc2D, Vector2i(PX, PY));

            // TODO: for large solid angles (in our case 8x8 is probably large)
            // we need to better approximate the BxDF single point sample of the material
            // should not cut it.
            // Further research on this can be approximating using Gaussian, cosine lobes etc.
            //
            // Fortunately it is used for path guiding and any error will convey towards the
            // variance instead of the result.
            Vector3f bxdfColored = warpSurf.Evaluate(wo, wi, medium);
            //if(rayIndex == 0)
            //{
            //    printf("WN:(%f, %f, %f), Wo(%f, %f, %f), Wi(%f, %f, %f), bxdf %f\n",
            //           warpSurf.WorldNormal()[0],
            //           warpSurf.WorldNormal()[1],
            //           warpSurf.WorldNormal()[2],
            //           wo[0], wo[1], wo[2],
            //           wi[0], wi[1], wi[2],
            //           Utility::RGBToLuminance(bxdfColored));
            //}

            // Convert it to single channel
            bxdfGray = Utility::RGBToLuminance(bxdfColored);
        }

        float radiance = (linearId < (PX * PY)) ?
                            shMem.sRadianceFieldSmall[loc2D[1]][loc2D[0]]
                            : 0.0f;
        productField[j] = max(MathConstants::LargeEpsilon, bxdfGray * radiance);
    }
}

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
template <class ProjFunc>
__device__ __forceinline__
void ProductSampler<TPB, X, Y, PX, PY>::GeneratePDFAndCDF(float(&pdfX)[REFL_PER_THREAD],
                                                          float(&cdfX)[REFL_PER_THREAD],
                                                          float& pdfY,
                                                          float& cdfY,
                                                          int32_t rayIndex,
                                                          ProjFunc&& Project) const
{
    // Multiply the outer field with the BXDF
    float productField[REFL_PER_THREAD];
    MultiplyWithBXDF(productField, rayIndex, Project);
    // Our field is generated now sample,
    // For first stage do a CDF parallel search search style
    // approach
    for(int32_t j = 0; j < REFL_PER_THREAD; j++)
    {
        float rowTotal;
        WarpRowScan(shMem.cubMem[warpId].rowScanMem).InclusiveSum(productField[j],
                                                                  cdfX[j], rowTotal);

        float rowTotalRecip = 1.0f / rowTotal;
        cdfX[j] *= rowTotalRecip;
        pdfX[j] = productField[j] * PX_FLOAT * rowTotalRecip;
        // CDF per lane is done
        // Store the total value for marginal pdf
        // rowLeader writes the data
        int32_t rowIndex = (warpLocalId / PX) + (j * PROCESSED_ROW_PER_ITER);
        bool isRowLeader = (warpLocalId % PX == 0);
        if(isRowLeader && rowIndex < PY)
            shMem.sWarpTempMemory[warpId][rowIndex] = rowTotal;
    }
    // Now calculate Marginal
    // Assuming PY fits in to a warp (it should be)
    pdfY = (warpLocalId < PY) ? shMem.sWarpTempMemory[warpId][warpLocalId] : 0.0f;
    float columnTotal;
    WarpColumnScan(shMem.cubMem[warpId].colScanMem).InclusiveSum(pdfY, cdfY,
                                                                 columnTotal);
    float columnTotalRecip = 1.0f / columnTotal;
    cdfY *= columnTotalRecip;
    pdfY *= PY_FLOAT * columnTotalRecip;

    if(isWarpLeader && columnTotal == 0.0f)
        printf("Entire product field is zero !\n");
}

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
__device__ __forceinline__
Vector2f ProductSampler<TPB, X, Y, PX, PY>::Sample(bool& innerIsUniform, float& pdfOut,
                                                   RNGeneratorGPUI& rng,
                                                   const float(&pdfX)[REFL_PER_THREAD],
                                                   const float(&cdfX)[REFL_PER_THREAD],
                                                   const float& pdfY,
                                                   const float& cdfY) const
{
    // Generated Marginal & Conditional Probabilities
    // Parallel search the CDFY
    float xi0 = (isWarpLeader) ? rng.Uniform() : 0.0f;
    xi0 = __shfl_sync(0xFFFFFFFF, xi0, 0, PY);
    // Check the votes
    uint32_t mask = __ballot_sync(0xFFFFFFFF, (xi0 > cdfY));
    int32_t rowId = __ffs(~mask) - 1;
    // Store the PDF Y
    if(warpLocalId == rowId)
        shMem.sWarpTempMemory[warpId][0] = pdfY;

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
    // Align the LSB of the row local result to actual LSB of the data
    mask >>= reducedRow * PX;
    // Calculate the column region that has the sample
    int columnId = __ffs(~mask) - 1;
    // Store the PDFX
    if(myColumnId == columnId && isMyRow)
        shMem.sWarpTempMemory[warpId][1] = pdfX[accessIndex];

    // Sanity Check
    assert((rowId < PY) && (columnId < PX));
    // Product sampling phase is done!
    // now subsample the inner small region (could be same size)
    //
    // Here we will use rejection sampling, it is highly parallel,
    // each warp will do n round of roll check. If we could not find
    // anything, warp leader will sample uniformly
    // Rejection sampling
    // https://www.realtimerendering.com/raytracinggems/rtg/index.html Chapter 16
    static constexpr Vector2f DPP_FLOAT(DATA_PER_PRODUCT[0], DATA_PER_PRODUCT[1]);
    static constexpr Vector2i DPP = DATA_PER_PRODUCT;
    Vector2i outerRegion2D = Vector2i(columnId, rowId);
    // Values that are found out by the rejection sampling
    Vector2f uv; float texVal;
    int32_t luckyWarp;
    static constexpr int REJECTION_SAMPLE_ITERATIONS = 16;
    for(int i = 0; i < REJECTION_SAMPLE_ITERATIONS; i++)
    {
        // Find a region
        uv = rng.Uniform2D();
        Vector2i innerRegion2D = Vector2i(uv * DPP_FLOAT);
        Vector2i index2D = outerRegion2D * DPP + innerRegion2D;
        // Sanity check
        assert((innerRegion2D[1] < DATA_PER_PRODUCT[1]) &&
               (innerRegion2D[0] < DATA_PER_PRODUCT[0]));

        // Rejection sample the found out region
        texVal = shMem.sRadianceField[index2D[1]][index2D[0]];
        float normalizedVal = texVal / shMem.sNormalizationConstants[outerRegion2D[1]][outerRegion2D[0]];
        mask = __ballot_sync(0xFFFFFFFF, (rng.Uniform() <= normalizedVal));
        luckyWarp = __ffs(mask) - 1;

        // Broadcast to the leader warp if a warp sampled the value.
        // (well; to everybody as well because of the to instruction nature)
        if(luckyWarp > 0)
        {
            uv[0] = __shfl_sync(0xFFFFFFFF, uv[0], luckyWarp, WARP_SIZE);
            uv[1] = __shfl_sync(0xFFFFFFFF, uv[1], luckyWarp, WARP_SIZE);
            texVal = __shfl_sync(0xFFFFFFFF, texVal, luckyWarp, WARP_SIZE);
        }
        if(luckyWarp != -1) break;
    }
    // Leader now does the remaining calculations and provide sampled UV & pdf
    if(isWarpLeader)
    {
        float integral = shMem.sRadianceFieldSmall[outerRegion2D[1]][outerRegion2D[0]];
        // Now calculate actual pdf etc.
        float pdfInner = (luckyWarp >= 0) ? texVal / integral : 1.0f;
        // Finally actual pdf
        pdfOut = shMem.sWarpTempMemory[warpId][0] *
                 shMem.sWarpTempMemory[warpId][1] * pdfInner;

        innerIsUniform = (luckyWarp < 0);
        if(luckyWarp < 0)
            printf("Unable to reject! Uniform Sampling "
                   "pdf(%f) = %f * %f * %f\n",
                   pdfOut, shMem.sWarpTempMemory[warpId][0],
                   shMem.sWarpTempMemory[warpId][1], pdfInner);
        if(isnan(pdfOut))
            printf("NaN pdf(%f) = %f * %f * %f\n",
                   pdfOut, shMem.sWarpTempMemory[warpId][0],
                   shMem.sWarpTempMemory[warpId][1], pdfInner);

        // Calculate the UV
        Vector2f innerRegionFloat = Vector2f(uv * DPP_FLOAT);
        Vector2f uvGuide = Vector2f(outerRegion2D) * DPP_FLOAT + innerRegionFloat;

        static constexpr Vector2f TOTAL_REGION_SIZE_RECIP = Vector2f(1.0f / X, 1.0f / Y);
        uvGuide *= TOTAL_REGION_SIZE_RECIP;
        // All done!
        return uvGuide;
    }
    // (Return is only valid for the warp leader(lane0))
    return Vector2f(NAN, NAN);
}

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
__device__ __forceinline__
float   ProductSampler<TPB, X, Y, PX, PY>::Pdf(const Vector2f& uv,
                                               const float(&pdfX)[REFL_PER_THREAD],
                                               const float pdfY,
                                               bool innerWasUniform) const
{
    static constexpr Vector2i DPP = DATA_PER_PRODUCT;
    // Calculate indices
    Vector2f outerExpanded = uv * Vector2f(PX_FLOAT, PY_FLOAT);
    Vector2f outerIndex = outerExpanded.Floor();
    Vector2f innerIndex = outerExpanded - outerIndex;
    Vector2i outerRegion2D = Vector2i(outerIndex);
    Vector2i innerRegion2D = Vector2i(innerIndex.Floor());
    Vector2i global2D = outerRegion2D * DPP + innerRegion2D;

    float innerIntegral = shMem.sRadianceFieldSmall[outerRegion2D[1]][outerRegion2D[0]];
    float innerValue = shMem.sRadianceField[global2D[1]][global2D[0]];
    // Now calculate actual pdf etc.
    float pdfInner = (innerWasUniform) ? 1.0f : innerValue / innerIntegral;

    // Shuffle the values for outer pdfY
    // Only valid between lanes [0, PY)
    // warpLeader will return so its fine
    float pdfMarginal = __shfl_sync(0xFFFFFFFF, pdfY, outerRegion2D[1], PY);

    // Each warp
    // Which index of the pdfX[] array are we need?
    int32_t innerLaneIndex = outerRegion2D[1] / PROCESSED_ROW_PER_ITER;
    // Which lane of that pdfX[] should we use
    int32_t outerIndexLinear = (outerRegion2D[1] * PX + outerRegion2D[1]);
    int32_t laneId = outerIndexLinear % WARP_SIZE;
    float pdfConditional = __shfl_sync(0xFFFFFFFF, pdfX[innerLaneIndex], laneId, WARP_SIZE);
    // Finally actual pdf
    return pdfConditional * pdfMarginal * pdfInner;
}

template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
template <class ProjFunc>
__device__ __forceinline__
Vector2f ProductSampler<TPB, X, Y, PX, PY>::SampleWithProduct(float& pdf,
                                                              RNGeneratorGPUI& rng,
                                                              int32_t rayIndex,
                                                              ProjFunc&& Project) const
{
    // Generate Fields (Functions)
    float pdfX[REFL_PER_THREAD];
    float cdfX[REFL_PER_THREAD];
    float pdfY;
    float cdfY;
    GeneratePDFAndCDF(pdfX, cdfX, pdfY, cdfY, rayIndex, Project);

    // Just Sample
    bool innerIsUniformSampled;
    Vector2f uv = Sample(innerIsUniformSampled, pdf,
                         rng,
                         pdfX, cdfX, pdfY, cdfY);
    return uv;
}


template <int32_t TPB, int32_t X, int32_t Y,
          int32_t PX, int32_t PY>
template <class ProjFunc, class InvProjFunc, class NormProjFunc>
__device__ __forceinline__
Vector2f ProductSampler<TPB, X, Y, PX, PY>::SampleMIS(float& pdf,
                                                      RNGeneratorGPUI& rng,
                                                      int32_t rayIndex,
                                                      ProjFunc&& Project,
                                                      InvProjFunc&& InvProject,
                                                      NormProjFunc&& NormProject,
                                                      float sampleRatio,
                                                      float projectionPdfMultiplier) const
{
    const GPUMetaSurface& warpSurf = shMem.sSurfaces[warpId];

    // Generate Fields (Functions)
    float pdfX[REFL_PER_THREAD];
    float cdfX[REFL_PER_THREAD];
    float pdfY;
    float cdfY;
    GeneratePDFAndCDF(pdfX, cdfX, pdfY, cdfY, rayIndex, Project);


    // Generate a sample and broadcast
    float xi = (isWarpLeader) ? rng.Uniform() : 0.0f;
    xi = __shfl_sync(0xFFFFFFFF, xi, 0, WARP_SIZE);

    // Before sampling check if it is needed
    static constexpr int32_t INVALID_RAY_INDEX = -1;
    bool invalidRay = (rayIndex == INVALID_RAY_INDEX);
    bool isLight = warpSurf.IsLight();
    bool isSpecular = (warpSurf.Specularity() >= TracerConstants::SPECULAR_THRESHOLD);

    if(isLight || isSpecular || invalidRay)
    {
        pdf = 1.0f;
        return Vector2f(0.0f);
    }

    if(isSpecular && isWarpLeader)
    {
        printf("spec %f\n", warpSurf.Specularity());
    }

    Vector2f sampledUV;
    float pdfSampled, pdfOther;
    if(xi >= sampleRatio)
    {
        // Sample BxDF (solo)
        //if(isWarpLeader)
        {
            const GPUMediumI* outMedium;
            RayF wo;
            Vector3f wi = -(metaSurfGenerator.Ray(gRayIds[rayIndex]).ray.getDirection());
            warpSurf.Sample(wo, pdfSampled,
                            outMedium,
                            //
                            wi,
                            GPUMediumVacuum(0),
                            rng);

            sampledUV = InvProject(wo.getDirection());
        }
        // Broadcast uv for pdf
        sampledUV[0] = __shfl_sync(0xFFFFFFFF,  sampledUV[0], 0, WARP_SIZE);
        sampledUV[1] = __shfl_sync(0xFFFFFFFF,  sampledUV[1], 0, WARP_SIZE);

        // PDF Guide (warp)
        pdfOther = Pdf(sampledUV, pdfX, pdfY, false);
        pdfOther *= projectionPdfMultiplier;
        pdfOther *= 2;
        sampleRatio = 1.0f - sampleRatio;

        //printf("[BxDF] pdfBxDF %f, pdfGuide %f\n", pdfSampled, pdfOther);
    }
    else
    {
        // Sample Guide (warp)
        bool innerIsUniformSampled;
        sampledUV = Sample(innerIsUniformSampled, pdfSampled,
                             rng, pdfX, cdfX, pdfY, cdfY);
        // Pdf BxDF (solo)
        //if(isWarpLeader)
        {
            pdfSampled *= projectionPdfMultiplier;
            pdfSampled *= 2;

            Vector3f wi = -(metaSurfGenerator.Ray(gRayIds[rayIndex]).ray.getDirection());
            Vector3f wo = NormProject(sampledUV);
            pdfOther = warpSurf.Pdf(wo, wi, GPUMediumVacuum(0));

            //printf("[Guide] pdfBxDF %f, pdfGuide %f\n", pdfSampled, pdfOther);
        }
    }

    if(isWarpLeader)
    {
        using namespace TracerFunctions;
        pdf = pdfSampled * sampleRatio / BalanceHeuristic(sampleRatio, pdfSampled,
                                                          1.0f - sampleRatio, pdfOther);
        pdf = (pdfSampled == 0.0f) ? 0.0f : pdf;

        //printf("pdf %f uv (%f, %f)\n", pdf, sampledUV[0], sampledUV[1]);
        return sampledUV;
    }
    return Vector2f(NAN, NAN);
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