#pragma once
/**

Static Interface of a BVH traversal
with custom Intersection and Hit acceptance

*/

#include "RayStructs.h"

class AABB3f
{

};

template <class LeafStruct, class PrimitiveData>
__device__ float(*IntersctionFunc)(const RayReg& r,
								   const LeafStruct& data,
								   const PrimitiveData& gPrimData);

template <class HitStruct>
__device__  bool (*AcceptHitFunc)(HitStruct& data, 
								  RayReg& r, 
								  float newT);

template <class PrimitiveData>
__device__  AABB3f(*BoxGenFunc)(uint32_t primitiveId, const PrimitiveData&);

template <class PrimitiveData>
__device__  float(*AreaGenFunc)(uint32_t primitiveId, const PrimitiveData&);

// Fundamental BVH Tree Node
template<class LeafStruct>
struct alignas(8) SceneNode
{
	static constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();

	// Pointers 
	uint64_t left;
	uint64_t right;			
	union
	{
		struct
		{
			Vector3 aabbMin;
			Vector3 aabbMax;
		};
		LeafStruct leaf;
	};
	uint64_t parent;
};


// This is fundemental BVH traversal kernel
// It supparts partial traversal and continuation traversal(for scene tree)
// It is customizable by intersection and hit determination
// Its output is customizable by HitStructs
template <class HitGMem, class HitReg,
		  class LeafStruct, class PrimitiveData,
		  IntersctionFunc<LeafStruct, PrimitiveData> IFunc,
		  AcceptHitFunc<HitReg> AFunc>
__global__ void KCIntersectBVH(RayGMem* rays,
							   HitGMem hits,
							   uint32_t rayCount,
							   // Constants
							   const SceneNode<LeafStruct>* gBVHList,
							   const PrimitiveData gPrimData,
							   const uint32_t* initalLoc)
{
	// Convenience Functions
	auto IsAlreadyTraversed = [](uint64_t list, uint32_t depth) -> bool
	{
		return ((list >> depth) & 0x1) == 1;
	};
	auto Pop = [](uint64_t& list, uint32_t& depth)
	{
		list += (1 << depth);
		depth++;
	};

	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; i += blockDim.x * gridDim.x)
	{
		// Load initial traverse point if available
		uint32_t startList = (initalLoc != nullptr) initalLoc[globalId] ? : 0;

		// Load Ray to Register
		RayReg ray(rays, globalId);
		HitReg hit(hits, globalId);

		// Depth First Search over BVH
		uint32_t depth = sizeof(uint64_t) * 8;
		BVHAcceleratorGPU::SceneNode<LeafStruct> currentNode = bvhList[0];
		for(uint64_t& list = startList; i < 0xFFFFFFFF;)
		{
			// Fast pop if both of the children is carries current node is zero
			// (This means that bit is carried)
			if(IsAlreadyTraversed(list, depth))
			{
				currentNode = bvhList[currentNode.parent];
				Pop(list, depth);
			}
			// Check if it is leaf node
			else if(currentNode.left == SceneNode<LeafStruct>::NULL_NODE &&
					currentNode.right == SceneNode<LeafStruct>::NULL_NODE)
			{
				// Do Intersection Test
				float newT = IFunc(ray, node.leaf, primitiveList);
				// Do Hit Acceptance break traversal if terminate is called
				if(AFunc(hit, ray, newT)) break;

				// Continue
				Pop(list, depth);
			}
			else if(ray.Intersects(currentNode.aabbMin,
								   currentNode.aabbMax))
			{
				// Continue Traversal (And check if we traversed left node already
				if(currentNode.left != SceneNode<LeafStruct>::NULL_NODE &&
				   !IsAlreadyTraversed(list, depth - 1))
				{
					currentNode = bvhList[currentNode.left];
					depth--;
					continue;
				}
				else if(currentNode.right != SceneNode<LeafStruct>::NULL_NODE)
				{
					currentNode = bvhList[currentNode.right];
					depth--;
					continue;
				}
			}
			else
			{
				// Skip Leafs
				currentNode = bvhList[currentNode.parent];
				Pop(list, depth);
			}
		}
			// Write Updated Stuff
			// Only tMax of ray can be changed
		ray.UpdateTMax(rays, globalId);
		hit.Update(hits, globalId);
	}
}

// This is fundamental BVH generation Kernels
// These can be implemented by custom aabb fetch functions


// Writes surface area of each primitive
// These will be reduced to an average value.
template <class PrimitiveData,
		  AreaGenFunc<PrimitiveData> AreaFunc>
__global__ void KCGeneratePrimitiveAreaBVH(float* gOutArea,
										   // Input
										   const PrimitiveData gPrimData,
										   const uint32_t primtiveCount)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < primtiveCount; i += blockDim.x * gridDim.x)
	{
		float area = AreaFunc(globalId, gPrimData);
		gOutArea[globalId] = area;
	}
}

// Determines how many cells should cover a primitive
template <class PrimitiveData,
		  BoxGenFunc<PrimitiveData> BoxFunc>
__global__ void KCDetermineCellCountBVH(uint32_t* gOutCount,
										// Input
										const PrimitiveData gPrimData,
										const uint32_t primtiveCount,
										const float optimalArea)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < primtiveCount; i += blockDim.x * gridDim.x)
	{
		AABB3f aabb = BoxFunc(globalId, gPrimData);

		// Using this and optimal area,
		// find total sub-primtivie count
		uint32_t cellCount = 0;
		// TODO: implement

		gOutCount[globalId] = cellCount;
	}
}

// Here do scan over primitive count for 

// Generates Partial AABB and morton numbers for each partial data
template <class PrimitiveData, BoxGenFunc<PrimitiveData> BoxFunc>
__global__ void KCGenerateParitalDataBVH(AABB3f* gSubAABBs,
										 uint64_t gMortonCodes,
										 // Input
										 const uint32_t* gPrimId,
										 const PrimitiveData gPrimData,
										 const uint32_t subPrimtiveCount,
										 const float optimalArea)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < subPrimtiveCount; i += blockDim.x * gridDim.x)
	{
		uint32_t primitiveId = gPrimId[globalId];
		AABB3f area = BoxFunc(primitiveId, gPrimData);

		// Using this and optimal area,
		// find total sub-primtivie count
		uint32_t cellCount = 0;
		// TODO: implement

		gOutCount[globalId] = cellCount;
	}
}

__global__ void KCPartialCount()
{

}


template <class LeafStruct, class PrimitiveData,
		  F<LeafStruct, PrimitiveData> 