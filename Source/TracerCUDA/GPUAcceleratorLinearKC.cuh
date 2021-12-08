#pragma once
/**

Static Interface of Loop based primitive traversal
with custom Intersection and Hit

*/

#include <array>

#include "RayLib/HitStructs.h"
#include "RayLib/SceneStructs.h"

#include "AcceleratorFunctions.h"
#include "GPUTransformIdentity.cuh"
#include "GPUAcceleratorCommonKC.cuh"
#include "CudaSystem.hpp"

#include <cub/cub.cuh>

// This is fundamental Linear traversal kernel
template <class PGroup>
__global__ CUDA_LAUNCH_BOUNDS_1D
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
                              const typename PGroup::LeafData* gLeafList,
                              const Vector2ul* gAccRanges,
                              const GPUTransformI** gTransforms,
                              const TransformId* gAccTransformIds,
                              PrimTransformType transformType,
                              //
                              const typename PGroup::PrimitiveData primData)
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
            HitResult result = AcceptHit<PGroup>(// Output
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
            ray.UpdateTMax(gRays, id);
            gHitStructs.Ref<HitData>(id) = hit;
            gTransformIds[id] = transformId;
            gMaterialKeys[id] = materialKey;
            gPrimitiveIds[id] = primitiveId;
        }
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
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