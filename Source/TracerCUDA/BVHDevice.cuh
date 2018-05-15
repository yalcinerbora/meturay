#pragma once
/**

Bounding Volume Hierarchy Implementation for GPU's
using CUDA.

Basic BVH implementation it is used to get things going with the tracer.
It can probably be improved a lot.


*/

#include <cstdint>
#include <vector>

#include "RayLib/Vector.h"
#include "RayLib/DeviceMemory.h"
#include "RayLib/Ray.h"

class MeshBatch;
struct Mesh;
struct ObjectAABBListDevice;
struct HitRecord;

class BVHDevice
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
		
		// Root of generated nodes
		DeviceMemory			nodes;
		Node*					root;

	protected:
	public:
		// Constructors & Destructor
										BVHDevice();
										BVHDevice(const BVHDevice&) = default;
										BVHDevice(BVHDevice&&) = default;
		BVHDevice&						operator=(const BVHDevice&) = default;
		BVHDevice&						operator=(BVHDevice&&) = default;
										~BVHDevice() = default;
		

		// Construction & Allocation
		__host__ void					DetermineAllocationSize(const MeshBatch&, const Mesh&);
		__host__ void					DetermineAllocationSize(const std::vector<Mesh>&);

		__device__ void					ConstructDevice(const MeshBatch&, const Mesh&);
		__device__ void					ConstructDevice(const std::vector<Mesh>&);		

		// Intersection
		__device__ bool					IntersectsFull(HitRecord&,
													   const RayF&) const;
		__device__ 	bool				IntersectsPartial(uint32_t& index,
														  HitRecord&,
														  const RayF&) const;
};
