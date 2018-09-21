#pragma once
/**

Static Interface of a BVH traversal
with custom Intersection and Hit acceptance

*/

#include "RayStructs.h"
#include "AABB.h"

// Intersection function is used to determine if a leaf node data;
// which is custom, is intersects with ray. Returns valid float if such hit exists
// return NAN otherwise.
// Primitive data is custom structure that holds gpu pointers or static data globally
// Leaf struct is custom structure that holds in the leaf.
// User then can determine outcome using both.
//
// (i.e. leafStruct holds triangle index and primitive index, Primitive data holds triangles
// of multiple primitives)
//
// (i.e. leafStruct holds parameters for a sphere with necessary inverse transformation, Primitive
// data holds nothing at all)
// etc.
template <class LeafStruct, class PrimitiveData>
__device__ float(*IntersctionFunc)(const RayReg& r,
								   const LeafStruct& data,
								   const PrimitiveData& gPrimData);

// Accept hit function is used to update hit structure of the ray
// It returns immidiate termination if necessary (i.e. when any hit is enough like
// in volume rendering).
template <class HitStruct>
__device__  bool (*AcceptHitFunc)(HitStruct& data, 
								  RayReg& r, 
								  float newT);

// Custom bounding box generation function for primitive
template <class PrimitiveData>
__device__  AABB3f(*BoxGenFunc)(uint32_t primitiveId, const PrimitiveData&);

// Surface area generation function for primitive
template <class PrimitiveData>
__device__  float(*AreaGenFunc)(uint32_t primitiveId, const PrimitiveData&);

// Fundamental BVH Tree Node
template<class LeafStruct>
struct alignas(8) BVHNode
{
	static constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();

	// Pointers	
	union
	{
		struct
		{			
			// 8 Word
			Vector3 aabbMin;
			uint32_t left;
			Vector3 aabbMax;
			uint32_t right;
			// 1 Word
			uint32_t parent;
		};
		LeafStruct leaf;
	};
	bool isLeaf;	
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
							   const BVHNode<LeafStruct>* gBVHList,
							   const PrimitiveData gPrimData,
							   const Vector2ui* gInitialListAndNode)
{
	// Convenience Functions
	auto IsAlreadyTraversed = [](uint64_t list, uint32_t depth) -> bool
	{
		return ((list >> depth) & 0x1) == 1;
	};
	auto MarkAsTraversed(uint64_t& list, uint32_t depth)
	{
		list += (1 << depth);
	};
	auto Pop = [](uint64_t& list, uint32_t& depth)
	{
		MarkAsTraversed(list, depth);
		depth++;
	};

	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; i += blockDim.x * gridDim.x)
	{
		// Load initial traverse point if available
		uint32_t startList = (gInitialListAndNode != nullptr) ? initalLoc[globalId][0] : 0;
		uint32_t initalLoc = (gInitialListAndNode != nullptr) ? initalLoc[globalId][1] : 0;		

		// Load Ray to Register
		RayReg ray(rays, globalId);
		HitReg hit(hits, globalId);

		// Depth First Search over BVH
		const uint32_t depth = sizeof(uint64_t) * 8;
		BVHNode<LeafStruct> currentNode = bvhList[initalLoc];
		for(uint64_t& list = startList; list < 0xFFFFFFFF;)
		{
			// Fast pop if both of the children is carries current node is zero
			// (This means that bit is carried)
			if(IsAlreadyTraversed(list, depth))
			{
				currentNode = bvhList[currentNode.parent];
				Pop(list, depth);
			}
			// Check if we already traversed left child
			// If child bit is on this means lower left child is traversed
			else if(IsAlreadyTraversed(list, depth - 1) &&
					currentNode.right != BVHNode<LeafStruct>::NULL_NODE)
			{
				// Go to right child
				currentNode = bvhList[currentNode.right];
				depth--;
			}
			// Now this means that we entered to this node first time
			// Check if this node is leaf or internal
			// Check if it is leaf node
			else if(isLeaf)
			{
				// Do Intersection Test
				float newT = IFunc(ray, node.leaf, primitiveList);
				// Do Hit Acceptance break traversal if terminate is called
				if(AFunc(hit, ray, newT)) break;

				// Continue
				Pop(list, depth);
			}
			// Not leaf so check AABB
			else if(ray.Intersects(currentNode.aabbMin, currentNode.aabbMax)
			{
				// Go left if avail
				if(currentNode.left != BVHNode<LeafStruct>::NULL_NODE)
				{
					currentNode = bvhList[currentNode.left];
					depth--;
				}
				// If not avail and since we are first time on this node
				// Try to go right
				else if(currentNode.left != BVHNode<LeafStruct>::NULL_NODE)
				{
					// In this case dont forget to mark left child as traversed
					MarkAsTraversed(list, depth - 1);

					currentNode = bvhList[currentNode.right];
					depth--;
				}
				else
				{
					// This should not happen
					// since we have "isNode" boolean
					assert(false);

					// Well in order to be correct mark this node traversed also
					// In the next iteration node will pop itself
					MarkAsTraversed(list, depth - 1);
				}
			}
			// Finally no ray is intersected
			// Go to parent
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

// Determines how many cells should a primitive cover
template <class PrimitiveData,
		  BoxGenFunc<PrimitiveData> BoxFunc,
		  AreaGenFunc<PrimitiveData> AreaFunc>
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
		// Generate your data
		AABB3f aabb = BoxFunc(globalId, gPrimData);
		float primitiveArea = AreaGenFunc(globalId, gPrimData);

		// Compare yoursef with area to generate triangles
		// Find how many splits are on you
		// Get limitation
		uint32_t splitCount;

		// TODO: implement

		gOutCount[globalId] = cellCount;
	}
}

// Here do scan over primitive count for 

// Generates Partial AABB and morton numbers for each partial data
template <class PrimitiveData, BoxGenFunc<PrimitiveData> BoxFunc>
__global__ void KCGenerateParitalDataBVH(AABB3f* gSubAABBs,
										 uint64_t* gMortonCodes,
										 // Input
										 const uint32_t* gPrimId,
										 const PrimitiveData gPrimData,
										 const uint32_t subPrimtiveCount,
										 const AABB3f& extents,
										 const float optimalArea)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < subPrimtiveCount; i += blockDim.x * gridDim.x)
	{
		// Generate parent data
		uint32_t parentId = gPrimId[globalId];
		AABB3f aabb = BoxFunc(parentId, gPrimData);
		float primitiveArea = AreaGenFunc(parentId, gPrimData);

		// Using parent primitive data and relative index
		// Find subAABB
		// And generate Morton code for that AABB centroid
		AABB3f subAABB = aabb; // TODO: calculate
		uint64_t morton = DiscretizePointMorton(subAABB.Centroid(),
												extents, optimalArea);

		gSubAABBs[globalId] = subAABB;
		gMortonCodes[globalId] = morton;
	}
}

// After that do a sort over morton codes
template <class LeafStruct, class PrimitiveData>
__global__ void KCGenerateBVH(BVHNode<LeafStruct>* gBVHList,
							  //
							  const uint32_t* gPrimId,							  
							  const uint64_t* gMortonCodes,

							  const uint32_t subPrimtiveCount)
{
	uint32_t internalNodeCount = subPrimtiveCount - 1;

	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < internalNodeCount; i += blockDim.x * gridDim.x)
	{
		// Binary Search


	}
}