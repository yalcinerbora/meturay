#pragma once

#include <cstdint>

#include "Vector.h"
#include "DeviceMemory.h"
#include "Ray.h"

class BVH
{
	private:
		enum NodeType : uint8_t
		{
			LEFT,
			RIGHT
		};

		struct alignas(16) Node
		{
			// Top down
			uint32_t			left;
			uint32_t			right;
			
			// For Backtracking without a stack
			uint32_t			parent;

			// 3 more word avail
			NodeType			nodeType;
			
			// For non-leaf AABB is used
			// If left and right is 0xFFFF..F
			// Then node is considered leaf
			// and triangle and object id is used to fetch triangle etc
			union
			{
				struct
				{
					// Either object or triangle depending on 
					// what BVH is this
					uint32_t	primitiveId;
					// 6 more words are avail
				};

				struct
				{
					Vector3		aabbMin;
					Vector3		aabbMax;
				};
			};
		};

		static_assert(sizeof(Node) == 48, "BVHNode size is not as specified.");
		

		//
		DeviceMemory			nodes;

		// List of generated nodes
		Node*					nodes;

	protected:
	public:
		// Constructors & Destructor
										BVH();
										BVH(const BVH&) = default;
										~BVH() = default;
		BVH&							operator=(const BVH&) = default;

		// Construction & Allocation
		__host__ void					ReDetermineAllocation(ObjectAABBList&);
		__host__ void					ReDetermineAllocation(VertexDataReference&);
		__device__ __host__	void		Construct(VertexDataReference&);
		__device__ __host__	void		Construct(ObjectAABBList&);

		// Intersection Related
		__device__ __host__	bool		IntersectsFull(float& distance,
													  const RayF&) const;
		__device__ __host__	bool		IntersectsPartial(uint32_t& index,
														  float& distance,
														  const RayF&) const;
};
