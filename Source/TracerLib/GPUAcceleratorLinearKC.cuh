#pragma once
/**

Static Interface of Loop based primitive traversal
with ustom Intersection and Hit

*/

#include <array>

#include "HitStructs.cuh"
#include "AcceleratorDeviceFunctions.h"

#include "RayLib/SceneStructs.h"

using HitKeyList = std::array<HitKey, SceneConstants::MaxSurfacePerAccelerator>;
using PrimitiveRangeList = std::array<Vector2ul, SceneConstants::MaxSurfacePerAccelerator>;

struct HKList
{
	const HitKey materialKeys[SceneConstants::MaxSurfacePerAccelerator];
};

struct PRList
{
	const Vector2ul primRanges[SceneConstants::MaxSurfacePerAccelerator];
};

// Fundamental Construction Kernel
template <class PGroup>
__global__
static void KCConstructLinear(// O
							  PGroup::LeafData* gLeafOut,

							  // Input
							  const Vector2ul* gAccRanges,
							  //const HitKeyList materialKeys,
							  HKList mkList,
							  PRList prList,
							  //const PrimitiveRangeList primRanges,
							  const PGroup::PrimitiveData primData,
							  const uint32_t leafIndex)
{
	// Fetch Types from Template Classes
	using LeafData = typename PGroup::LeafData;	// LeafStruct is defined by primitive

	// Your Data
	const Vector2ul accRange = gAccRanges[leafIndex];
	const uint32_t leafCount = static_cast<const uint32_t>(accRange[1]);
	LeafData* gAccLeafs = gLeafOut + accRange[0];

	// SceneConstants
	uint32_t RangeLocation[SceneConstants::MaxSurfacePerAccelerator];
	
	auto FindIndex = [&](uint32_t globalId) -> int
	{
		static constexpr int LastLocation = SceneConstants::MaxSurfacePerAccelerator - 1;
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
	for(int i = 0; i < SceneConstants::MaxSurfacePerAccelerator; i++)
	{
		uint32_t primCount = static_cast<uint32_t>(prList.primRanges[i][1] - 
												   prList.primRanges[i][0]);
		totalPrimCount += primCount;
		RangeLocation[i] = totalPrimCount;
	}

	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < leafCount; globalId += blockDim.x * gridDim.x)
	{
		// Find index of range
		const uint32_t pairIndex = FindIndex(globalId);
		const uint32_t localIndex = globalId - RangeLocation[pairIndex];

		// Determine 
		uint64_t primitiveId = prList.primRanges[pairIndex][0] + localIndex;
		HitKey matKey = mkList.materialKeys[pairIndex];
		// Gen Leaf and write
		gAccLeafs[globalId] = PGroup::LeafFunc(matKey,
											   primitiveId,
											   primData);
	}
}

// This is fundemental Linear traversal kernel
template <class PGroup>
__global__
static void KCIntersectLinear(// O
							  HitKey* gMaterialKeys,
							  PrimitiveId* gPrimitiveIds,
							  HitStructPtr gHitStructs,
							  // I-O
							  RayGMem* gRays,
							  // Input
							  const TransformId* gTransformIds,
							  const RayId* gRayIds,
							  const HitKey* gAccelKeys,
							  const uint32_t rayCount,
							  // Constants
							  const PGroup::LeafData* gLeafList,
							  const Vector2ul* gAccRanges,
							  const TransformStruct* gInverseTransforms,
							  //
							  const PGroup::PrimitiveData primData)
{
	// Fetch Types from Template Classes
	using HitData = typename PGroup::HitData;		// HitRegister is defined by primitive
	using LeafData = typename PGroup::LeafData;		// LeafStruct is defined by primitive

	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; globalId += blockDim.x * gridDim.x)
	{
		const uint32_t id = gRayIds[globalId];
		const uint64_t accId = HitKey::FetchIdPortion(gAccelKeys[globalId]);
		const TransformId transformId = gTransformIds[id];

		// Load Ray/Hit to Register
		RayReg ray(gRays, id);
		//HitReg hit(gHitStructs, id);
		//HitId hitId = gHitKeys[id];

		// Key is the index of the inner Linear Array
		const Vector2ul accRange = gAccRanges[accId];
		const LeafData* gLeaf = gLeafList + accRange[0];
		const uint32_t endCount = static_cast<uint32_t>(accRange[1]);

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
		HitData hit;

		// Linear check over array
		for(uint32_t i = 0; i < endCount; i++)
		{
			// Get Leaf Data and
			// Do acceptance check
			const LeafData leaf = gLeaf[i];
			HitResult result = PGroup::HitFunc(// Ooutput											 
											   materialKey,
											   primitiveId,
											   hit,
											   // I-O
											   ray,
											   // Input
											   leaf,
											   primData);
			hitModified = result[1];
			if(result[0]) break;
		}
		// Write Updated Stuff
		if(hitModified)
		{
			ray.UpdateTMax(gRays, globalId);
			gHitStructs.Ref<HitData>(globalId) = hit;
			gMaterialKeys[id] = materialKey;
			gPrimitiveIds[id] = primitiveId;
		}
	}
}


__global__
static void KCIntersectBaseLinear(// I-O
								  TransformId* gTransformIds,
								  HitKey* gHitKeys,
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
		globalId < rayCount; globalId += blockDim.x * gridDim.x)
	{
		const uint32_t id = gRayIds[globalId];
		
		// Load Ray to Register
		RayReg ray(gRays, id);	

		// Load initial traverse location
		uint32_t primStart = gPrevLoc[id];

		// Check next potential hit		
		HitKey key = HitKey::InvalidKey;
		TransformId transformId = 0;
		while(primStart < leafCount)
		{
			BaseLeaf l = gLeafs[primStart];
			primStart++;
			if(ray.ray.IntersectsAABB(l.aabbMin, l.aabbMax))
			{
				key = l.accKey;
				transformId = l.transformId;				
				break;
			}			
		}
		// Write next potential hit
		if(primStart < leafCount)
		{
			// Write Updated Stuff
			gPrevLoc[id] = primStart;
			gHitKeys[globalId] = key;
			gTransformIds[id] = transformId;
		}
	}
}