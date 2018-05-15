#pragma once
/**

Sparse Voxel Octree Implementation for GPU's using CUDA.

Basic SVO implementation used to accelerate GPUVolume intersection
and traversal.


*/

#include <cstdint>

#include "RayLib/RayHitStructs.h"
#include "RayLib/DeviceMemory.h"
#include "RayLib/Vector.h"

struct HitRecord;
class VolumeI;

class SVODevice
{
	private:	
		struct Node
		{
			uint32_t				next;
			uint32_t				parent;
		};

		// Extents
		Vector3						aabbMin;
		Vector3						aabbMax;

		Vector3ui					dimensions;
		uint32_t					totalLevel;

	protected:
		DeviceMemory				 mem;
		
		// Ptrs
		Node*						root;


	public:
		// Constructors & Destructor
										SVODevice();
										SVODevice(const SVODevice&) = default;
										SVODevice(SVODevice&&) = default;
		SVODevice&						operator=(const SVODevice&) = default;
		SVODevice&						operator=(SVODevice&&) = default;
										~SVODevice() = default;

		// Construction & Allocation
		void							DetermineAllocationSize(const Vector3ui& volumeSize);
		void							ConstructDevice(const Vector3ui& volumeSize, const VolumeI&);

		// Intersection
		__device__ bool					Intersects(HitRecord&, const RayF&) const;
};


__device__ 
inline bool SVODevice::Intersects(HitRecord& record, const RayF& ray) const
{
	Vector3 pos;
	float t, tTotal = 0;

	// Initially check if ray is outside of the box
	// And advance it to the nearest edge + epsilon
	RayF currentRay = ray;
	if(currentRay.IntersectsAABB(pos, t, aabbMin, aabbMax) &&
	   pos[0] == aabbMin[0] ||
	   pos[1] == aabbMin[1] || 
	   pos[2] == aabbMin[2])
	{
		currentRay.AdvanceSelf(t + MathConstants::Epsilon);
		tTotal += (t + MathConstants::Epsilon);
	}

	// Traverse through volume until ray is outside
	// Or code will break out if it hits
	while(currentRay.IntersectsAABB(pos, t, aabbMin, aabbMax))
	{
		// Find relative index of position
		Vector3f location = ray.getPosition();
		Vector3ui index = ;
		
		// Traverse through tree for that location
		// and find its aabb				
		Node* n = root;
		for(uint32_t depth = 0; depth <= totalLevel; depth++)
		{
			uint32_t childLoc;
			//....

			if(n == nullptr) break;
			n = root + n->next + childLoc;
		}

		// Generate Traversed AABB
		Vector3f localAABBMin = ;
		Vector3f localAABBMax = ;

		// Do an intersection with that level of the AABB
		// This should always enter (by definition)
		if(currentRay.IntersectsAABB(pos, t, localAABBMin, localAABBMax))
		{
			currentRay.AdvanceSelf(t + MathConstants::Epsilon);
			tTotal += (t + MathConstants::Epsilon);
		}
		
		// Chcek if we found a node
		if(n->next == 0xFFFFFFFE)
		{
			// Float index of location
			record.baryCoord = ;
			record.distance = tTotal;
			
			record.objectId = 123;
			record.triangleId = 123;

			return true;
		}
	}
	return false;
}