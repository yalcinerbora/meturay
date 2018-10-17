#pragma once
/**

Static Interface of Loop based primitive traversal
with ustom Intersection and Hit

*/


#include "HitStructs.cuh"
#include "AcceleratorDeviceFunctions.h"

#include "RayLib/SceneStructs.h"

// This is fundemental Linear traversal kernel

//template <class HitGMem, class HitReg,
//		  class LeafStruct, class PrimitiveData,
//		  IntersctionFunc<LeafStruct, PrimitiveData> IFunc,
//		  AcceptHitFunc<HitReg> AFunc>
template <class PGroup>
__global__ void KCIntersectLinear(// O
								  HitKey* gMaterialKeys,
								  PrimitiveId* gPrimitiveIds,
								  HitStructPtr gHitStructs,
								  // I-O
								  RayGMem* gRays,								  
								  // Input
								  const TransformId* dTransformIds,
								  const RayId* gRayIds,
								  const HitKey* gHitKeys,
								  const uint32_t rayCount,
								  // Constants
								  const PGroup::LeafStruct** gLeafList,
								  const uint32_t* gLeafCounts,
								  const TransformStruct* gInverseTransforms,
								  const PGroup::PrimitiveData primData)
{
	// Fetch Types from Template Classes
	using HitReg = typename PGroup::HitReg;			// HitRegister is defined by primitive
	using LeafStruct = typename PGroup::LeafStruct;	// LeafStruct is defined by primitive

	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; globalId += blockDim.x * gridDim.x)
	{
		const uint32_t id = gRayIds[globalId];
		const uint64_t accId = HitConstants::FetchIdMask(gHitKeys[globalId]);
		const TransformId transformId = dTransformIds[id];

		// Load Ray/Hit to Register
		RayReg ray(gRays, id);
		//HitReg hit(gHitStructs, id);
		//HitId hitId = gHitKeys[id];

		// Key is the index of the inner Linear Array
		const LeafStruct* gLeaf = gLeafList[accId];
		const uint32_t* gEndCount = gLeafCounts[accId];

		// Zero means identity so skip
		if(transformId != 0)
		{
			TransformStruct s = gInverseTransforms[transformId];
			ray.ray.TransformSelf(s);
		}	


		
		// Hit determination
		bool hitModified = false;
		HitKey materialKey;
		PrimitiveId primitiveId;
		HitReg hit;

		// Linear check over array
		for(uint32_t i = 0; i < gEndCount; i++)
		{
			// Get Leaf Data
			// 
			const LeafStruct leaf = gLeaf[i];			
			HitResult result = PGroup::AFunc(// Ooutput											 
											 materialKey,
											 primitiveId,
											 hit,
											 // I-O
											 ray, 
											 // Input
											 primData,
											 leaf);
			hitModified = result[1];
			if(result[0]) break;
		}
		// Write Updated Stuff
		if(hitModified)
		{
			ray.UpdateTMax(ray, globalId);
			hit.Update(gHitStructs, globalId);
			gMaterialKeys[id] = materialKey;
			gPrimitiveIds[id] = primitiveId;
		}
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