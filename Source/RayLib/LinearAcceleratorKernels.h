#pragma once
/**

Static Interface of Loop based primitive traversal
with ustom Intersection and Hit

*/

#include "HitStructs.h"

#include "AcceleratorDeviceFunctions.h"

// This is fundemental Linear traversal kernel

template <class HitGMem, class HitReg,
		  class LeafStruct, class PrimitiveData,
		  IntersctionFunc<LeafStruct, PrimitiveData> IFunc,
		  AcceptHitFunc<HitReg> AFunc>
__global__ void KCIntersectLinear(RayGMem* gRays,
								  HitGMem* gHits,
								  const RayId* gIds,
								  uint32_t rayCount,
								  // Constants
								  const PrimitiveData gPrimData,
								  const Vector2ui* gPrimStartEnd)

{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; i += blockDim.x * gridDim.x)
	{
		// Load Ray/Hit to Register
		RayReg ray(rays, globalId);
		HitReg hit(hits, globalId);

		// Linear check over array
		for(uint32_t i = gPrimStartEnd[0]; i < gPrimStartEnd[1]; i++)
		{
			// Do Intersection Test
			float newT = IntersctionFunc(r, LeafStruct{i}, gPrimData);
			if(AFunc(hit, ray, newT))
			{
				ray.UpdateTMax(rays, globalId);
				hit.Update(hits, globalId);
				break;
			}
		}
		// Write Updated Stuff
		// Only tMax of ray which could have changed
		ray.UpdateTMax(rays, globalId);
		hit.Update(hits, globalId);
	}
}


template <class LeafStruct, class PrimitiveData>
__global__ void KCNextIntersectLinear(// Outputs
									  HitKey* dKeys,	
									  // I-O
									  uint32_t* gPrevLocation,
									  // Inputs
									  const RayGMem* gRays,
									  const RayId* gIds,
									  uint32_t rayCount,
									  // Constants
									  const PrimitiveData gPrimData,
									  const Vector2ui* gPrimStartEnd)
{
		// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; i += blockDim.x * gridDim.x)
	{		
		// Load Ray/Hit to Register
		RayId id = gIds[globalId]
		RayReg ray(rays, id);
		HitReg hit(hits, id);

		// Linear check over array
		for(uint32_t i = gPrevLocation; i < gPrimStartEnd[1]; i++)
		{
			//

			// Do Intersection Test
			float newT = IntersctionFunc(r, LeafStruct{i}, gPrimData);
			if(AFunc(hit, ray, newT))
			{
				ray.UpdateTMax(rays, globalId);
				hit.Update(hits, globalId);
				break;
			}
		}
		// Write Updated Stuff
		// Only tMax of ray which could have changed
		ray.UpdateTMax(rays, id);
		hit.Update(hits, id);
	}
}