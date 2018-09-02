#pragma once
/**

Accelerator Customizable GPU Structure

This has shader like behaviour but it is quite limited.
Intersection funct returns a distance based on logic of the mesh
It can return closest intersection


*/

#include <RayLib/RayStructs.h>
#include <RayLib/GPUCode.h>



// Fundamental Leafs
struct LowerLeaf
{
	uint32_t	materialId;
	uint32_t	primitiveId;
	uint32_t	objectId;
};

struct UpperLeaf
{
	uint32_t	acceleratorId;
};

template <class LeafStruct, class PrimitiveList>
float(*IntersctionFunc)(const RayReg& r,
						const LeafStruct& data,
						const PrimitiveList* gPrimitives);

template <class HitStruct>
bool (*AcceptHitFunc)(HitStruct& data, RayReg& r, float newT);

class AcceleratorGPU : public GPUCode
{
	private:
		// Launches Kernel
		virtual void							LaunchKernel(RayGMem* rays,
															 uint32_t rayCount,
															 uint32_t acceleratorId) = 0;

};

class BVHAcceleratorGPU : public AcceleratorGPU
{
	public:
		static constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();

		template<class LeafStruct>
		struct alignas(8) SceneNode
		{
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

		 
	private:	
	protected:
	public:
		// Constructors & Destructor


		
};

/**

This is fundemental BVH traversal shader
It supparts partial traversal and continuation traversal (for scene tree)
It is customizable by intersection and hit determination
Its output is customizable by HitStructs

*/
template <class HitGMem, class HitReg,
		  class LeafStruct, class PrimitiveList,
		  IntersctionFunc<LeafStruct, PrimitiveList> IFunc,
		  AcceptHitFunc<HitReg> AFunc>
__global__ void BVHIntersection(RayGMem* rays,
								HitGMem hits,
								uint32_t rayCount,
								// Constants
								const BVHAcceleratorGPU::SceneNode<LeafStruct>* bvhList,
								const PrimitiveList& primitiveList)
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

	int a = 10'000'000;


	// GlobalId determination
	uint32_t globalId = 0;

	// Load Ray to Register
	RayReg ray(rays, globalId);
	HitReg hit(hits, globalId);

	// Depth First Search over BVH
	uint32_t depth = sizeof(uint64_t) * 8;
	BVHAcceleratorGPU<LeafStruct>::SceneNode currentNode = bvhList[0];
	for(uint64_t& list = hit.list; i < 0xFFFFFFFF;)
	{
		// Fast pop if both of the children is carries current node is zero
		// (This means that bit is carried)
		if(IsAlreadyTraversed(list, depth))
		{
			currentNode = bvhList[currentNode.parent];
			Pop(list, depth);
		}
		// Check if it is leaf node
		else if(currentNode.left == BVHAcceleratorGPU::NULL_NODE &&
				currentNode.right == BVHAcceleratorGPU::NULL_NODE)
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
			if(currentNode.left != BVHAcceleratorGPU::NULL_NODE &&
			   !IsAlreadyTraversed(list, depth - 1))
			{
				currentNode = bvhList[currentNode.left];
				depth--;
				continue;
			}
			else if(currentNode.right != BVHAcceleratorGPU::NULL_NODE)
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
		// Write Updated Stuff
		// Only tMax of ray can be changed
		ray.UpdateTMax(rays, globalId);
		hit.Update(hits, globalId);
}