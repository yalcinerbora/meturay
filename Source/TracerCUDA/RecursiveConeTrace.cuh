#pragma once

#include "AnisoSVO.cuh"

template <class T, class WrapFunctor>
class Span2D
{
    private:
    T*                  memory;
    Vector2i            size;
    const WrapFunctor&  wrapFunc;

    public:
    // Constructor
    __device__ __forceinline__
    Span2D(const WrapFunctor& wf, T* memory, Vector2i sz)
        : memory(memory)
        , size(sz)
        , wrapFunc(wf)
    {}

    // I-O
    __device__ __forceinline__
    float& operator()(int x, int y)
    {
        Vector2i xy = wrapFunc(Vector2i(x, y), size);
        return memory[xy[1] * size[0] + xy[0]];
    }

    __device__ __forceinline__
    const float& operator()(int x, int y) const
    {
        Vector2i xy = wrapFunc(Vector2i(x, y), size);
        return memory[xy[1] * size[0] + xy[0]];
    }
};

template <int32_t TPB, int32_t X, int32_t Y>
class BatchConeTracer
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

    public:
    static constexpr int32_t DATA_PER_THREAD = std::max(1, (X * Y) / TPB);
    struct TempStorage
    {
        float sTMinBuffer[(X >> 1) * (Y >> 1)];
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
                                              Vector3f(&rayDirOut)[DATA_PER_THREAD],
                                              float(&tMinOut)[DATA_PER_THREAD],
                                              bool(&isLeaf)[DATA_PER_THREAD],
                                              uint32_t(&nodeIndex)[DATA_PER_THREAD],
                                              // Inputs Common
                                              const Vector3f& rayPos,
                                              const Vector2f& tMinMaxInit,
                                              float pixelSolidAngle,
                                              int32_t recursionDepth,
                                              int32_t recursionJump,
                                              uint32_t maxQueryLevelOffset,
                                              // Functors
                                              RayProjectFunc&& ProjFunc,
                                              ProjWrapFunc&& WrapFunc);

    template<class RayProjectFunc>
    __device__ void     BatchedConeTraceRay(//Outputs
                                            Vector3f(&rayDirOut)[DATA_PER_THREAD],
                                            float(&tMinOut)[DATA_PER_THREAD],
                                            bool(&isLeaf)[DATA_PER_THREAD],
                                            uint32_t(&nodeIndex)[DATA_PER_THREAD],
                                            // Inputs Common
                                            const Vector3f& rayPos,
                                            const Vector2f& tMinMax,
                                            float pixelSolidAngle,
                                            uint32_t maxQueryLevelOffset,
                                            // Functors
                                            RayProjectFunc&& ProjFunc);
};

template <int32_t TPB, int32_t X, int32_t Y>
__device__ __forceinline__
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
                                                       Vector3f(&rayDirOut)[DATA_PER_THREAD],
                                                       float(&tMinOut)[DATA_PER_THREAD],
                                                       bool(&isLeaf)[DATA_PER_THREAD],
                                                       uint32_t(&nodeIndex)[DATA_PER_THREAD],
                                                       // Inputs Common
                                                       const Vector3f& rayPos,
                                                       const Vector2f& tMinMaxInit,
                                                       float pixelSolidAngle,
                                                       int32_t recursionDepth,
                                                       int32_t recursionJump,
                                                       uint32_t maxQueryLevelOffset,
                                                       // Functors
                                                       RayProjectFunc&& ProjFunc,
                                                       ProjWrapFunc&& WrapFunc)
{
    auto FindBottomLeft = [](const Vector2i& id, int32_t recursionJump)
    {
        return Vector2i(((id[0] + recursionJump) >> recursionJump) - 1,
                        ((id[1] + recursionJump) >> recursionJump) - 1);
    };

    #pragma unroll
    for(int i = 0; i < DATA_PER_THREAD; i++)
        tMinOut[i] = tMinMaxInit[0];

    const bool isSingleTrace = (recursionDepth == 0);

    // recursion jump
    recursionJump = max(1, recursionJump);
    recursionDepth *= recursionJump;
    for(int32_t r = recursionDepth; r >= 0; r -= recursionJump)
    {
        // Current iteration region data
        Vector2i currentRegion = Vector2i(X >> r, Y >> r);
        int32_t currentTotalWork = currentRegion.Multiply();
        int32_t dataPerThread = max(1, (currentTotalWork) / TPB);
        // Mask the linear array according to current region size
        Span2D tMinBuffer(WrapFunc, sMem.sTMinBuffer, currentRegion);
        // Launch Cone Trace Rays for the region
        for(int i = 0; i < dataPerThread; i++)
        {
            int32_t localId = i * TPB + threadId;
            if(localId >= currentTotalWork) continue;

            Vector2i rayId(localId % currentRegion[0],
                           localId / currentRegion[0]);
            // Generate Ray
            rayDirOut[i] = ProjFunc(rayId, currentRegion);
            RayF ray(rayDirOut[i], rayPos);

            float currentSolidAngle = pixelSolidAngle;
            // Conservatively expand the solid angle of this iteration
            // solid angle grows by factor of 4 also multiply it accordingly.
            // add extra padding to make sure it covers every inner pixel
            //static constexpr float Factor = 3.0f / 2.0f * MathConstants::Sqrt2;
            float solidAngleMultiplier = static_cast<float>(1 << (r << 2));// *Factor;
            if(r != 0) currentSolidAngle *=  solidAngleMultiplier;

            tMinOut[i] = svo.ConeTraceRay(isLeaf[i], nodeIndex[i], ray,
                                          tMinOut[i], tMinMaxInit[1],
                                          currentSolidAngle, maxQueryLevelOffset);
            // Write the found out tMin (except on the last iteration)
            if(r != 0) tMinBuffer(rayId[0], rayId[1]) = tMinOut[i];
        }

        // Single trace mode just exit
        if(isSingleTrace) break;

        // Make sure everybody written to the shared mem
        __syncthreads();
        // Change the thread logic, load the tMin of the threads for next iteration
        // Current iteration region data
        int32_t nextR = r - recursionJump;
        Vector2i nextRegion = Vector2i(X >> nextR, Y >> nextR);
        int32_t nextTotalWork = nextRegion.Multiply();
        int32_t nextDataPerThread = max(1, (nextTotalWork) / TPB);
        // Next iteration's thread will fetch the tMins from the current buffer
        for(int i = 0; i < nextDataPerThread; i++)
        {
            int32_t localId = i * TPB + threadId;
            if(localId >= nextTotalWork) continue;

            // This rays rayId
            Vector2i nextRayId(localId % nextRegion[0],
                               localId / nextRegion[0]);
            // Generate previous tMinBuffer
            Vector2i oldBottomLeft = FindBottomLeft(nextRayId, recursionJump);
            Vector4f tMins(tMinBuffer(oldBottomLeft[0], oldBottomLeft[1]),
                           tMinBuffer(oldBottomLeft[0] + 1, oldBottomLeft[1]),
                           tMinBuffer(oldBottomLeft[0], oldBottomLeft[1] + 1),
                           tMinBuffer(oldBottomLeft[0] + 1, oldBottomLeft[1] + 1));
            // Conservatively estimate the next iteration tMin
            tMinOut[i] = tMins[tMins.Min()];
        }
        // Make sure all threads read a tMin
        __syncthreads();
    }
    // On last iteration found out stuff is written to proper locations
    // All Done!
}

template <int32_t TPB, int32_t X, int32_t Y>
template<class RayProjectFunc>
__device__ __forceinline__
void BatchConeTracer<TPB, X, Y>::BatchedConeTraceRay(//Outputs
                                                     Vector3f(&rayDirOut)[DATA_PER_THREAD],
                                                     float(&tMinOut)[DATA_PER_THREAD],
                                                     bool(&isLeaf)[DATA_PER_THREAD],
                                                     uint32_t(&nodeIndex)[DATA_PER_THREAD],
                                                     // Inputs Common
                                                     const Vector3f& rayPos,
                                                     const Vector2f& tMinMax,
                                                     float pixelSolidAngle,
                                                     uint32_t maxQueryLevelOffset,
                                                     // Functors
                                                     RayProjectFunc&& ProjFunc)
{
    // Launch Cone Trace Rays for the region
    for(int i = 0; i < DATA_PER_THREAD; i++)
    {
        int32_t localId = i * TPB + threadId;
        if(localId >= X * Y) continue;

        Vector2i rayId(localId % X, localId / X);
        // Generate Ray
        rayDirOut[i] = ProjFunc(rayId, Vector2i(X, Y));
        RayF ray(rayDirOut[i], rayPos);

        tMinOut[i] = svo.ConeTraceRay(isLeaf[i], nodeIndex[i], ray,
                                      tMinMax[0], tMinMax[1],
                                      pixelSolidAngle, maxQueryLevelOffset);
    }
}