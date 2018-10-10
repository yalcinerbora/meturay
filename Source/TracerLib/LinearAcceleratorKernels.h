#pragma once
/**

Static Interface of Loop based primitive traversal
with ustom Intersection and Hit

*/


#include "HitStructs.h"
#include "AcceleratorDeviceFunctions.h"

#include "RayLib/SceneStructs.h"

// This is fundemental Linear traversal kernel

template <class HitGMem, class HitReg,
		  class LeafStruct, class PrimitiveData,
		  IntersctionFunc<LeafStruct, PrimitiveData> IFunc,
		  AcceptHitFunc<HitReg> AFunc>
	__global__ void KCIntersectLinear(// I-O
									  RayGMem* gRays,
									  HitGMem* gHits,
									  HitId* gHitIds,
									  // Input
									  const TransformId* dTransformIds,
									  const RayId* gRayIds,
									  const HitKey* gHitKeys,
									  const uint32_t rayCount,
									  // Constants
									  const LeafStruct** gLeafList,
									  const uint32_t* gEndCountList,
									  const TransformStruct* gInverseTransforms,
									  const PrimitiveData gPrimData)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; globalId += blockDim.x * gridDim.x)
	{
		const uint32_t id = gRayIds[globalId];
		const uint64_t accId = HitConstants::FetchIdMask(gHitKeys[globalId]);
		const TransformId transformId = dTransformIds[id];

		// Load Ray/Hit to Register
		RayReg ray(gRays, id);
		HitReg hit(gHits, id);
		HitId hitId = gHitIds[id];

		// Key is the index of the inner Linear Array
		const LeafStruct* gLeaf = gLeafList[accId];
		const uint32_t* gEndCount = gEndCountList[accId];

		// Zero means identity so skip
		if(transformId != 0)
		{
			TransformStruct s = gInverseTransforms[transformId];
			ray.ray.TransformSelf(s);
		}	

		// Linear check over array
		for(uint32_t i = 0; i < gEndCount; i++)
		{
			// Do Intersection

			// Accept Hit

			// Do Intersection Test
			float newT = IFunc(r, gLeaf[i], gPrimData);
			if(AFunc(hit, hitId, ray, newT)) break;
			
		}
		// Write Updated Stuff
		ray.UpdateTMax(rays, globalId);
		hit.Update(hits, globalId);
		gHitIds[id] = hitId;
	}
}


__global__ void KCIntersectBaseLinear(// I-O
									  TransformId* gTransformIds,
									  HitKey* gHitKeys,
									  uint32_t* gPrevLoc,
									  // Input
									  const RayGMem* gRays,
									  const RayId* gRayIds,
									  const uint32_t rayCount,

									  // Constants
									  const BaseLeaf* gKeys)
{
		// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; globalId += blockDim.x * gridDim.x)
	{
		const uint32_t id = gRayIds[globalId];
		
		// Load Ray/Hit to Register
		RayReg ray(gRays, id);
		HitKey key = gHitKeys[globalId];				
		TransformId transformId = gTransformIds[id];		
		if(key == HitConstants::InvalidKey) continue;

		// Load initial traverse extentds
		uint32_t primStart = gPrevLoc[id];

		// Check next
		key = gKeys[primStart].accKey;
		transformId = gKeys->transformId;
		
		primStart++;

		// Write Updated Stuff
		gPrevLoc[id] = primStart;
		gHitKeys[globalId] = key;
		gTransformIds[id] = transformId;
	}
}