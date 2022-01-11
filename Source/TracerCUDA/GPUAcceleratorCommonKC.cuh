#pragma once

#include "RayLib/AABB.h"
#include "GPUTransformI.h"
#include "GPUPiecewiseDistribution.cuh"
#include "AcceleratorFunctions.h"

struct HKList
{
    const HitKey materialKeys[SceneConstants::MaxPrimitivePerSurface];
};

struct PRList
{
    const Vector2ul primRanges[SceneConstants::MaxPrimitivePerSurface];
};

// Accept hit function
// Return two booleans first boolean tells control flow to terminate
// intersection checking (when finding any hit is enough),
// other boolean is returns that the hit is accepted or not.
//
// If hit is accepted Accept hit function should return
// valid material key, primitiveId, and hit.
//
// Material Key is the index of material
// Primitive id is the index of the individual primitive
// Hit is the interpolating weights of the primitive
//
// PrimitiveData struct holds the array of the primitive data
// (normal, position etc..)
template <class PGroup>
__device__ __forceinline__
static HitResult AcceptHit(// Output
                           HitKey& newMat,
                           PrimitiveId& newPrim,
                           typename PGroup::HitData& newHit,
                           // I-O
                           RayReg& rayData,
                           // Input
                           const GPUTransformI& transform,
                           const typename PGroup::LeafData& leaf,
                           const typename PGroup::PrimitiveData& primData)
{
    using HitData = typename PGroup::HitData;

    float newT;
    HitData hitData;
    bool intersects = PGroup::Intersects(// Output
                                         newT,
                                         hitData,
                                         // I-O
                                         rayData,
                                         // Input
                                         transform,
                                         leaf,
                                         primData);
    bool closerHit = intersects && (newT < rayData.tMax);
    // If intersected do alpha test as well
    if(closerHit)
    {
        bool opaque = PGroup::AlphaTest(hitData, leaf, primData);
        closerHit &= opaque;
    }

    // Save the hit if it is closer
    if(closerHit)
    {
        rayData.tMax = newT;
        newMat = leaf.matId;
        newPrim = leaf.primitiveId;
        newHit = hitData;
    }
    return HitResult{false, closerHit};
}

// UnionAABB Reduce Functor
__device__ __forceinline__
AABB3f ReduceAABB3f(const AABB3f& aabb0,
                    const AABB3f& aabb1)
{
    return aabb0.Union(aabb1);
}

template <class PGroup>
__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCInitializeLeafs(// O
                              typename PGroup::LeafData* gLeafOut,
                              // Input
                              const Vector2ul* gAccRanges,
                              const HKList mkList,
                              const PRList prList,
                              const typename PGroup::PrimitiveData primData,
                              const uint32_t accIndex,
                              bool doKeyExpansion)
{
    // Fetch Types from Template Classes
    using LeafData = typename PGroup::LeafData; // LeafStruct is defined by primitive

    // Your Data
    const Vector2ul accRange = gAccRanges[accIndex];
    const uint32_t leafCount = static_cast<uint32_t>(accRange[1]-accRange[0]);
    LeafData* gAccLeafs = gLeafOut + accRange[0];

    // SceneConstants
    uint32_t RangeLocation[SceneConstants::MaxPrimitivePerSurface];

    auto FindIndex = [&](uint32_t globalId) -> int
    {
        static constexpr int LastLocation = SceneConstants::MaxPrimitivePerSurface - 1;
        #pragma unroll
        for(int i = 0; i < LastLocation; i++)
        {
            //
            if(globalId >= RangeLocation[i] &&
               globalId < RangeLocation[i + 1])
                return i;
        }
        return LastLocation;
    };

    // Initialize Offsets
    uint32_t totalPrimCount = 0;
    #pragma unroll
    for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
    {
        RangeLocation[i] = totalPrimCount;
        uint32_t primCount = static_cast<uint32_t>(prList.primRanges[i][1] -
                                                   prList.primRanges[i][0]);
        totalPrimCount += primCount;
    }

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < leafCount; globalId += blockDim.x * gridDim.x)
    {
        // Find index of range
        const uint32_t pairIndex = FindIndex(globalId);
        const uint32_t localIndex = globalId - RangeLocation[pairIndex];

        // Determine  Prim Id and Hit Key
        PrimitiveId primitiveId = prList.primRanges[pairIndex][0] + localIndex;
        HitKey matKey = mkList.materialKeys[pairIndex];
        if(doKeyExpansion)
        {
            PrimitiveId expandedId = HitKey::FetchIdPortion(matKey) + localIndex;
            matKey = HitKey::CombinedKey(HitKey::FetchBatchPortion(matKey),
                                         static_cast<HitKey::Type>(expandedId));
        }
        // Gen Leaf and write
        gAccLeafs[globalId] = PGroup::Leaf(matKey,
                                           primitiveId,
                                           primData);
    }
}

__global__
static void KCInitIndices(// O
                          uint32_t* gIndices,
                          PrimitiveId* gPrimIds,
                          // Input
                          uint32_t indexStart,
                          uint64_t rangeStart,
                          uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        gIndices[globalId] = indexStart + globalId;
        gPrimIds[globalId] = rangeStart + globalId;
    }
}

__global__
static void KCTransformAABB(AABB3f& gAABB,
                            const GPUTransformI& transform,
                            const AABB3f localAABB)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0)
    {
        gAABB = transform.LocalToWorld(localAABB);
    }
}

__global__
static void KCAcquireTransform(const GPUTransformI** gTransformPtr,
                               const uint32_t* gAccTransformIdList,
                               const GPUTransformI** gTransformList,
                               uint32_t innerIndex)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0)
    {
        *gTransformPtr = gTransformList[gAccTransformIdList[innerIndex]];
    }
}

__global__
static void KCAcquireTransform(const GPUTransformI** gTransformPtr,
                               const GPUTransformI** gTransformList,
                               uint32_t index)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0)
    {
        *gTransformPtr = gTransformList[index];
    }
}

__host__
inline void TransformLocalAABBToWorld(AABB3f& aabb, const GPUTransformI& transform,
                                      const CudaGPU& gpu)
{
    DeviceMemory tempMem(sizeof(AABB3f));
    // Generate World Space AABB from Local AABB
    gpu.KC_X(0, 0, 1,
             KCTransformAABB,
             *static_cast<AABB3f*>(tempMem),
             transform,
             aabb);

    AABB3f aabbReturn;
    CUDA_CHECK(cudaMemcpy(&aabbReturn, tempMem, sizeof(AABB3f),
                          cudaMemcpyDeviceToHost));
    aabb = aabbReturn;
}

__host__
inline void AcquireAcceleratorGPUTransform(const GPUTransformI*& transform,
                                           const uint32_t* dAccTransformIdList,
                                           const GPUTransformI** dTransformList,
                                           uint32_t innerIndex,
                                           const CudaGPU& gpu)
{
    DeviceMemory tempMem(sizeof(GPUTransformI*));
    // Generate World Space AABB from Local AABB
    gpu.KC_X(0, 0, 1,
             //
             static_cast<void (*)(const GPUTransformI**, const uint32_t*,
                                  const GPUTransformI**, uint32_t)>
             (KCAcquireTransform),
             //
             static_cast<const GPUTransformI**>(tempMem),
             dAccTransformIdList,
             dTransformList,
             innerIndex);

    GPUTransformI* ptr;
    CUDA_CHECK(cudaMemcpy(&ptr, tempMem, sizeof(GPUTransformI*),
                          cudaMemcpyDeviceToHost));
    transform = ptr;
}

__host__
inline void AcquireIdentityTransform(const GPUTransformI*& transform,
                                     const GPUTransformI** dTransformList,
                                     uint32_t identityTransformIndex,
                                     const CudaGPU& gpu)
{
    DeviceMemory tempMem(sizeof(GPUTransformI*));
    // Generate World Space AABB from Local AABB
    gpu.KC_X(0, 0, 1,
             //
             static_cast<void (*)(const GPUTransformI**, const GPUTransformI**,
                                  uint32_t)>
             (KCAcquireTransform),
             //
             static_cast<const GPUTransformI**>(tempMem),
             dTransformList,
             identityTransformIndex);

    GPUTransformI* ptr;
    CUDA_CHECK(cudaMemcpy(&ptr, tempMem, sizeof(GPUTransformI*),
                          cudaMemcpyDeviceToHost));
    transform = ptr;
}

__global__
static void KCInitPrimIdsAndIndices(// O
                                    uint32_t* gIndices,
                                    PrimitiveId* gPrimIds,
                                    // Input
                                    uint32_t indexStart,
                                    uint64_t rangeStart,
                                    uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        gIndices[globalId] = indexStart + globalId;
        gPrimIds[globalId] = rangeStart + globalId;
    }
}

template <class PGroup>
__global__
static void KCGenCentersWithIndex(// O
                                  Vector3f* gCenters,
                                  // Input
                                  const uint32_t* gIndicies,
                                  const PrimitiveId* gPrimitiveIds,
                                  //
                                  CentroidGen<PGroup> centFunc,
                                  uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        uint32_t id = gIndicies[globalId];
        PrimitiveId primId = gPrimitiveIds[id];

        gCenters[globalId] = centFunc(primId);
    }
}

template <class PGroup>
__global__
static void KCGenAABBsWithIndex(// O
                                AABB3f* gAABBs,
                                // Input
                                const uint32_t* gIndicies,
                                const PrimitiveId* gPrimitiveIds,
                                //
                                AABBGen<PGroup> aabbFunc,
                                uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        uint32_t id = gIndicies[globalId];
        PrimitiveId primId = gPrimitiveIds[id];

        gAABBs[globalId] = aabbFunc(primId);
    }
}

template <class PGroup>
__global__
static void KCGenAABBs(// O
                       AABB3f* gAABBs,
                       // Input
                       Vector2ul primRange,
                       //
                       AABBGen<PGroup> aabbFunc,
                       uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        PrimitiveId primId = primRange[0] + globalId;
        gAABBs[globalId] = aabbFunc(primId);
    }
}

template <class PGroup>
__global__
static void KCGenerateAreas(// Output
                            float* dAreas,
                            // Inputs
                            const typename PGroup::LeafData* gLeafs,
                            const TransformId* gTransformIds,
                            const GPUTransformI** gTransforms,
                            // Constants
                            const typename PGroup::PrimitiveData primData,
                            size_t totalPrimCount)
{
    static constexpr auto AreaFunc = PGroup::Area;

    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < totalPrimCount; globalId += blockDim.x * gridDim.x)
    {
        float area = AreaFunc(gLeafs[globalId].primitiveId, primData);
        // Scale the area
        const GPUTransformI* t = gTransforms[gTransformIds[globalId]];
        Vector3f scaleFactor = t->ToWorldScale();

        // Assume a scale matrix with this factors
        // Determinant should give approx scale
        float det = scaleFactor.Multiply();
        dAreas[globalId] = area * det;

        //printf("Scale:[%f, %f, %f], = %f, A %f\n",
        //       scaleFactor[0], scaleFactor[1], scaleFactor[2],
        //       det, area);
    }
}

template <class PGroup, class RNG>
__global__
static void KCSampleSurfacePatch(// Output
                                 Vector3f* dPositions,
                                 Vector3f* dNormals,
                                 // I-O
                                 RNGeneratorGPUI** gRNGs,
                                 // Inputs
                                 const typename PGroup::LeafData* gLeafs,
                                 const TransformId* gTransformIds,
                                 const GPUTransformI** gTransforms,
                                 // Constants
                                 GPUDistPiecewiseConst1D areaDist,
                                 const typename PGroup::PrimitiveData primData,
                                 size_t totalSampleCount)
{
    static constexpr auto SamplePos = PGroup::SamplePosition;

    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);

    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < totalSampleCount; globalId += blockDim.x * gridDim.x)
    {
        float pdf;
        float index;
        //areaDist.Sample(pdf, index, rng);
        index = 0.1f;
        uint32_t leafIndex = static_cast<uint32_t>(index);
        PrimitiveId primId = gLeafs[leafIndex].primitiveId;

        assert(leafIndex < areaDist.Count());

        const GPUTransformI& transform = *gTransforms[gTransformIds[leafIndex]];

        Vector3f normal;
        Vector3f pos = SamplePos(normal, pdf, transform,
                                 primId, primData, rng);

        //printf("TID %u\n", gTransformIds[leafIndex]);
        //printf("[%u] P:[%f, %f, %f], N[%f, %f, %f]\n",
        //       leafIndex,
        //       pos[0], pos[1], pos[2],
        //       normal[0], normal[1], normal[2]);

        dPositions[globalId] = pos;
        dNormals[globalId] = normal;
    }
}
