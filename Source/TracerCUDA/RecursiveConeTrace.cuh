#pragma once

#include "AnisoSVO.cuh"

template <int32_t TPB, int32_t X, int32_t Y>
class BatchConeTracer
{
    private:
    static constexpr bool TPBCheck(int32_t TPB, int32_t X, int32_t Y)
    {
        auto PIX_COUNT = (X * Y);
        if(TPB > PIX_COUNT) return TPB % PIX_COUNT == 0;
        if(TPB <= PIX_COUNT) return PIX_COUNT % TPB == 0;
        return false;
    }
    // No SFINAE, just static assert
    static_assert(TPBCheck(TPB, X, Y),
                  "TBP and (X * Y) must be divisible, (X*Y) / TBP or TBP / (X*Y)");

    public:
    static constexpr int32_t DATA_PER_THREAD = std::max(1u, (X * Y) / TPB);
    struct TempStorage
    {
        float sTMinBuffer[X * Y];
    };

    private:
    TempStorage&                sMem;
    const AnisoSVOctreeGPU&     svo;
    const int32_t               threadId;

    public:
    // Constructor
    __device__          BatchConeTracer(TempStorage& sMem,
                                        const AnisoSVOctreeGPU& svo);

    template<class RayProjectFunc, class ProjWrapFunc>
    __device__ void     RecursiveConeTraceRay(//Outputs
                                              float(&tMinOut)[DATA_PER_THREAD],
                                              float(&isLeaf)[DATA_PER_THREAD],
                                              float(&nodeIndex)[DATA_PER_THREAD],
                                              // Inputs Common
                                              const Vector3f& rayPos,
                                              float tMin,
                                              float tMax,
                                              float pixelSolidAngle,
                                              int32_t recursionDepth,
                                              // Functors
                                              const RayProjectFunc& ProjFunc,
                                              const ProjWrapFunc& WrapFunc);
};

template <int32_t TPB, int32_t X, int32_t Y>
__device__ inline
BatchConeTracer<TPB, X, Y>::BatchConeTracer(TempStorage& storage,
                                            const AnisoSVOctreeGPU& svo)
    : sMem(storage)
    , svo(svo)
    , threadId(threadIdx.x)
{}

template <int32_t TPB, int32_t X, int32_t Y>
template<class RayProjectFunc, class ProjWrapFunc>
__device__ inline
void BatchConeTracer<TPB, X, Y>::RecursiveConeTraceRay(//Outputs
                                                        float (&tMinOut)[DATA_PER_THREAD],
                                                        bool (&isLeaf)[DATA_PER_THREAD],
                                                        uint32_t (&nodeIndex)[DATA_PER_THREAD],
                                                        // Inputs Common
                                                        const Vector3f& rayPos,
                                                        float tMin,
                                                        float tMax,
                                                        float pixelSolidAngle,
                                                        int32_t recursionDepth,
                                                        // Functors
                                                        const RayProjectFunc& ProjFunc,
                                                        const ProjWrapFunc& WrapFunc)
{
    #pragma unroll
    for(int i = 0; i < DATA_PER_THREAD; i++)
        tMinOut[i] = tMin;

    for(int32_t i = recursionDepth; i >= 0; i--)
    {
        // Current iteration region data
        Vector2i currentRegion = Vector2i(regionSize[0] >> i,
                                          regionSize[1] >> i);
        int32_t currentTotalWork = currentRegion.Multiply();
        // Before writing to the system find out your tMin
        int32_t dataPerThread = max(1, (currentTotalWork) / TPB);
        for(int i = 0; i < dataPerThread; i++)
        {
            int32_t localId = i * TPB + threadId;
            if(localId >= currentTotalWork) continue;
            // Do not fetch from the shared memory the first iteration
            // We already set the currentTMin to initial tMin
            if(i == recursionDepth) continue;

            // This rays rayId
            Vector2i rayId(localId % currentRegion[0],
                           localId / currentRegion[0]);
            // Generate previous tMinBuffer
            Vector2i previousRegion = Vector2i(regionSize[0] >> (i + 1),
                                                regionSize[1] >> (i + 1));
            Span2D tMinBuffer(WrapFunc, sTMinBuffer, previousRegion);
            Vector2i oldBottomLeft = Vector2i((rayId[0] + 1) >> 1,
                                                (rayId[1] + 1) >> 1);
            Vector4f tMins(tMinBuffer(oldBottomLeft[0], oldBottomLeft[1]),
                            tMinBuffer(oldBottomLeft[0] + 1, oldBottomLeft[1]),
                            tMinBuffer(oldBottomLeft[0], oldBottomLeft[1] + 1),
                            tMinBuffer(oldBottomLeft[0] + 1, oldBottomLeft[1] + 1));
            // Conservatively estimate the next iteration tMin
            tMinOut[i] = tMins[tMins.Min()];
        }
        // Make sure all threads read a tMin
        __syncthreads();
        // Launch Cone Trace Rays for the region
        int32_t dataPerThread = max(1, (currentTotalWork) / TPB);
        for(int i = 0; i < dataPerThread; i++)
        {
            int32_t localId = i * TPB + threadId;
            if(localId >= currentTotalWork) continue;

            Vector2i rayId(localId % currentRegion[0],
                           localId / currentRegion[0]);

            // Generate current iteration data
            float currentSolidAngle = pixelSolidAngle * (1 << i);
            Span2D tMinBuffer(WrapFunc, sTMinBuffer, currentRegion);
            // Generate Ray
            Vector3f rayDir = ProjFunc(rayId, currentRegion);
            RayF ray(rayDir, rayPos);

            tMinOut[i] = svo.ConeTraceRay(isLeaf[i], nodeIndex[i], ray,
                                          tMinOut[i], tMax,
                                          currentSolidAngle, 0);
            // Write the found out tMin
            tMinBuffer(rayId[0], rayId[1]) = tMinOut[i];
        }
        __syncthreads();
    }
    // On last iteration found out stuff is written to proper locations
    // All Done!
}