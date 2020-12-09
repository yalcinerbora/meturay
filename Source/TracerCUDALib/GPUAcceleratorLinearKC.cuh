#pragma once
/**

Static Interface of Loop based primitive traversal
with ustom Intersection and Hit

*/

#include <array>

#include "RayLib/HitStructs.h"
#include "RayLib/SceneStructs.h"

#include "AcceleratorFunctions.h"
#include "GPUTransformIdentity.cuh"
#include "CudaConstants.hpp"

#pragma warning( push )
#pragma warning( disable : 4834)
#include <cub/cub.cuh>
#pragma warning( pop )

struct HKList
{
    const HitKey materialKeys[SceneConstants::MaxPrimitivePerSurface];
};

struct PRList
{
    const Vector2ul primRanges[SceneConstants::MaxPrimitivePerSurface];
};

template <class PGroup>
__global__
void KCGenAABBs(// O
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

// Fundamental Construction Kernel
template <class PGroup>
__global__ __launch_bounds__(StaticThreadPerBlock1D)
static void KCConstructLinear(// O
                              PGroup::LeafData* gLeafOut,
                              // Input
                              const Vector2ul* gAccRanges,
                              const HKList mkList,
                              const PRList prList,
                              const PGroup::PrimitiveData primData,
                              const uint32_t accIndex)
{
    // Fetch Types from Template Classes
    using LeafData = typename PGroup::LeafData; // LeafStruct is defined by primitive

    // Your Data
    const Vector2ul accRange = gAccRanges[accIndex];
    const uint32_t leafCount = static_cast<const uint32_t>(accRange[1]-accRange[0]);
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

        // Gen Leaf and write
        gAccLeafs[globalId] = PGroup::Leaf(matKey,
                                           primitiveId,
                                           primData);
    }
}

// This is fundemental Linear traversal kernel
template <class PGroup>
__global__ //__launch_bounds__(StaticThreadPerBlock1D)
static void KCIntersectLinear(// O
                              HitKey* gMaterialKeys,
                              TransformId* gTransformIds,
                              PrimitiveId* gPrimitiveIds,
                              HitStructPtr gHitStructs,
                              // I-O
                              RayGMem* gRays,
                              // Input
                              const RayId* gRayIds,
                              const HitKey* gAccelKeys,
                              const uint32_t rayCount,
                              // Constants
                              const PGroup::LeafData* gLeafList,
                              const Vector2ul* gAccRanges,
                              const GPUTransformI** gTransforms,
                              const TransformId* gAccTransformIds,
                              PrimTransformType transformType,
                              //
                              const PGroup::PrimitiveData primData)
{
    // Fetch Types from Template Classes
    using HitData = typename PGroup::HitData;       // HitRegister is defined by primitive
    using LeafData = typename PGroup::LeafData;     // LeafStruct is defined by primitive

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; globalId += blockDim.x * gridDim.x)
    {
        const uint32_t id = gRayIds[globalId];
        const uint64_t accId = HitKey::FetchIdPortion(gAccelKeys[globalId]);
        const TransformId transformId = gAccTransformIds[accId];

        // Load Ray to Register
        RayReg ray(gRays, id);
        
        // Key is the index of the inner Linear Array
        const Vector2ul accRange = gAccRanges[accId];
        const LeafData* gLeaf = gLeafList + accRange[0];
        const uint32_t endCount = static_cast<uint32_t>(accRange[1] - accRange[0]);

        // Check transforms
        GPUTransformIdentity identityTransform;
        const GPUTransformI* localTransform = &identityTransform;
        if(transformType == PrimTransformType::CONSTANT_LOCAL_TRANSFORM)
        {
            const GPUTransformI& t = (*gTransforms[transformId]);
            ray.ray = t.WorldToLocal(ray.ray);
        }
        else localTransform = gTransforms[transformId];

        // Hit determination
        bool hitModified = false;
        HitKey materialKey;
        PrimitiveId primitiveId;
        HitData hit;
        // Linear check over array
        for(uint32_t i = 0; i < endCount; i++)
        {
            // Get Leaf Data and
            // Do acceptance check
            const LeafData leaf = gLeaf[i];
            //printf("my accId 0x%X, matId %x, range{%llu, %llu}, \n", 
            //       gAccelKeys[globalId],
            //       leaf.matId.value,
            //       accRange[0], accRange[1]);

            HitResult result = PGroup::Hit(// Output                                            
                                           materialKey,
                                           primitiveId,
                                           hit,
                                           // I-O
                                           ray,
                                           // Input
                                           *localTransform,
                                           leaf,
                                           primData);
            hitModified |= result[1];
            if(result[0]) break;
        }
        // Write Updated Stuff
        if(hitModified)
        {
            //if(id == 95) printf("MatFound %x\n", materialKey.value);
            ray.UpdateTMax(gRays, id);
            gHitStructs.Ref<HitData>(id) = hit;
            gTransformIds[id] = transformId;
            gMaterialKeys[id] = materialKey;
            gPrimitiveIds[id] = primitiveId;
        }
    }
}


__global__ __launch_bounds__(StaticThreadPerBlock1D)
static void KCIntersectBaseLinear(// Output                                  
                                  HitKey* gHitKeys,
                                  // I-O
                                  uint32_t* gPrevLoc,
                                  // Input
                                  const RayGMem* gRays,
                                  const RayId* gRayIds,
                                  const uint32_t rayCount,
                                  // Constants
                                  const BaseLeaf* gLeafs,
                                  const uint32_t leafCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; 
        globalId += blockDim.x * gridDim.x)
    {
        const uint32_t id = gRayIds[globalId];
        
        // Load Ray to Register
        RayReg rayData(gRays, id);
        Vector2f tMinMax(rayData.tMin, rayData.tMax);
        RayF& ray = rayData.ray;

        // Load initial traverse location
        uint32_t primStart = gPrevLoc[id];
        primStart = rayData.IsInvalidRay() ? leafCount : primStart;
        // Check next potential hit     
        HitKey nextAccKey = HitKey::InvalidKey;
        for(; primStart < leafCount; primStart++)
        {
            BaseLeaf l = gLeafs[primStart];
            if(ray.IntersectsAABB(l.aabbMin, l.aabbMax, tMinMax))
            {
                //printf("Found Intersection %u, prev %u, Key %X\n", primStart,
                //       gPrevLoc[id], l.accKey.value);
                nextAccKey = l.accKey;
                break;
            }
        }

        // Write next potential hit
        if(primStart < leafCount)
        {            
            // Set AcceleratorId and TransformId for lower accelerator
            gHitKeys[globalId] = nextAccKey;
            // Save State for next iteration
            gPrevLoc[id] = primStart + 1;
            
        }
        // If we are out of bounds write invalid key
        else gHitKeys[globalId] = HitKey::InvalidKey;
    }
}