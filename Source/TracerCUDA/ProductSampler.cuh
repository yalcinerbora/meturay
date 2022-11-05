#pragma once

#include "RayLib/Types.h"
#include "RayLib/ColorConversion.h"

#include "GPUMetaSurfaceGenerator.h"
#include "GPUMetaSurface.h"
#include "GPUMediumVacuum.cuh"
#include "TracerConstants.h"

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

    static constexpr float PX_FLOAT = static_cast<float>(PX);
    static constexpr float PY_FLOAT = static_cast<float>(PY);

    struct SharedStorage
    {
        // Scratch pad for each warp
        //float sLocalProducts[WARP_PER_BLOCK][PRODUCT_MAP_SIZE[0]][PRODUCT_MAP_SIZE[1]];
        // Main radiance field
        float sRadianceField[X][Y];
        // Reduced radiance field (will be multiplied by the BxDF field)
        float sRadianceFieldSmall[PRODUCT_MAP_SIZE[0]][PRODUCT_MAP_SIZE[1]];

        // Temp Storage
        float sPDFY[WARP_PER_BLOCK][PY];

        //GPUMetaSurface sSurfaces[WARP_PER_BLOCK];
        Byte sSurfacesRaw[WARP_PER_BLOCK * sizeof(GPUMetaSurface)];
    };

    private:
    SharedStorage&                          shMem;

    const RayId*                            gRayIds;
    const GPUMetaSurfaceGeneratorGroup&     metaSurfGenerator;
    uint32_t                                rayCount;

    // Parallelization Logic Related
    int32_t                                 warpLocalId;
    int32_t                                 warpId;
    bool                                    isWarpLeader;
    bool                                    isRowLeader;

    public:
    __device__ ProductSampler(SharedStorage& sharedMem,
                              const RayId* gRayIds,
                              const GPUMetaSurfaceGeneratorGroup& generator)
        : shMem(sharedMem)
        , gRayIds(gRayIds)
        , metaSurfGenerator(generator)
        , warpLocalId(threadIdx.x % WARP_SIZE)
        , warpId(threadIdx.x / WARP_SIZE)
        , isWarpLeader(warpLocalId == 0)
        , isRowLeader((warpLocalId % PX) == 0)
    {}

    __device__
    Vector2f SampleProduct(uint32_t rayIndex,
                           float& pdf,
                           RNGeneratorGPUI& rng) const
    {
        // Convert Raw MetaSurface memory
        // TODO: bad design change this later
        GPUMetaSurface* sSurfaces = reinterpret_cast<GPUMetaSurface*>(shMem.sSurfacesRaw);

        // Load material & surface to shared mem
        // Call copy construct here (GPUMetaSurface has reference)
        //if(isWarpLeader)
        //    new (sSurfaces + warpId) GPUMetaSurface(metaSurfGenerator.AcquireWork(gRayIds[rayIndex]));

        // No need to sync here, threads are in lockstep in warp
        const GPUMetaSurface& surface = sSurfaces[warpId];
        // TODO: Change this to a specific medium, current is does not work
        //const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);
        GPUMediumVacuum medium(0);

        // Generate PX by PY product field (store it on register space)
        float productField[REFL_PER_THREAD];
        //for(int32_t i = warpId; i < (PX * PY); i += WARP_SIZE)
        for(int32_t j = 0; j < REFL_PER_THREAD; j++)
        {
            int x = warpId % PX;
            int y = warpId % PY;

            // Only calculate if surface is not specular
            Vector3f bxdfColored = Vector3f(1.0f);
            if(surface.Specularity() < TracerConstants::SPECULAR_THRESHOLD)
            {
                //bxdfColored = surface.Evaluate(Vector3f(1.0f),
                //                               Vector3f(1.0f),
                //                               medium);
                bxdfColored = Vector3f(static_cast<float>(i));
            }

            productField[i] = (Utility::RGBToLuminance(bxdfColored) *
                               shMem.sRadianceFieldSmall[y][x]);
        }

        // Our field is generated now sample,
        // For first stage do a CDF parallel search search style
        // approach
        float cdfX[REFL_PER_THREAD];
        float pdfX[REFL_PER_THREAD];
        for(int32_t j = 0; j < REFL_PER_THREAD; j++)
        {
            float value = productField[j];
            for(int i = 1; i <= (PX >> 1); i *= 2)
            {
                float n = __shfl_up_sync(0xFFFFFFFF, value, i, PX);
                if((warpLocalId & (PX - 1)) >= i) value += n;
            }
            // Broadcast the max value locals on sub-wrap basis for normalization total
            float rowTotal = __shfl_sync(0xFFFFFFFF, value, PX - 1, PX);

            float rowTotalRecip = 1.0f / rowTotal;
            cdfX[j] = value * rowTotalRecip;
            pdfX[j] = productField[j] * PX_FLOAT * rowTotalRecip;
            // CDF per lane is done
            // Store the total value for marginal pdf
            // rowLeader writes the data
            int32_t rowIndex = (warpLocalId / PX) + (j * PROCESSED_ROW_PER_ITER);
            if(warpLocalId % PX == 0) shMem.sPDFY[warpId][rowIndex] = rowTotal;
        }

        // Now calculate Marginal
        // Assuming py fits in to a warp (it should be)
        float value = (warpLocalId < PY) ? shMem.sPDFY[warpId][warpLocalId] : 0.0f;

        for(int i = 1; i <= (PY >> 1); i *= 2)
        {
            float n = __shfl_up_sync(0xFFFFFFFF, value, i, PY);
            if((warpLocalId & (PY - 1)) >= i) value += n;
        }
        // Broadcast the max value for normalization
        float columnTotal = __shfl_sync(0xFFFFFFFF, value, PY - 1, PY);
        float columnTotalRecip = 1.0f / columnTotal;

        float cdfY = value * columnTotalRecip;
        float pdfY = (warpLocalId < PY) ? shMem.sPDFY[warpId][warpLocalId] : 0.0f;
        pdfY *= PY_FLOAT * columnTotalRecip;

        // Generated Marginal & Conditional Probabilities
        // Parallel search the CDFY
        float xi0 = (isWarpLeader) ? rng.Uniform() : 0.0f;
        xi0 = __shfl_sync(0xFFFFFFFF, xi0, 0, PY);

        uint32_t mask = __match_any_sync(0xFFFFFFFF, (xi0 >= cdfY));
        int32_t rowId = __clz(mask);

        // save the xi_Y, (cdfY_0, cdfY_1]
        if(isWarpLeader)
            shMem.sPDFY[warpId][0] = xi0;
        if(warpLocalId == rowId)
        {
            shMem.sPDFY[warpId][1] = cdfY;
            shMem.sPDFY[warpId][3] = pdfY;
        }
        if(warpLocalId == rowId + 1)
            shMem.sPDFY[warpId][2] = cdfY;

        // Now do the same but for row
        int32_t myColumnId = (warpLocalId % PX);
        int32_t myRowIndex = (warpLocalId / PX);
        int32_t reducedRow = rowId % PROCESSED_ROW_PER_ITER;
        int accessIndex = rowId / PROCESSED_ROW_PER_ITER;
        bool isMyRow = (myRowIndex == reducedRow);

        // Broadcast the random number (for dimension X)
        float xi1 = (isRowLeader && isMyRow) ? rng.Uniform() : 0.0f;
        xi1 = __shfl_sync(0xFFFFFFFF, xi1, 0, PX);

        mask = __match_any_sync(0xFFFFFFFF, (xi1 >= cdfX[accessIndex]));
        int columnId = __clz(mask);
        // save the xi_X, (cdfX_0, cdfX_1]
        if(isMyRow && isRowLeader)
            shMem.sPDFY[warpId][4] = xi1;
        if(isMyRow && myColumnId == columnId)
        {
            shMem.sPDFY[warpId][5] = cdfX[accessIndex];
            shMem.sPDFY[warpId][7] = pdfX[accessIndex];
        }
        if(isMyRow && myColumnId == columnId + 1)
            shMem.sPDFY[warpId][6] = cdfX[accessIndex];

        // Product sampling phase is done!
        // now subsample the inner small region
    }
};