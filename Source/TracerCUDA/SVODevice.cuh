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
		static constexpr uint32_t	DATA_LEAF = 0xFFFFFFFE;
		static constexpr uint32_t	NULL_CHILD = 0xFFFFFFFF;

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

		uint32_t					volumeId;

		// Helper Functions
		__device__ static uint32_t	CalculateLevelChildId(const Vector3i& voxelPos,
														  const unsigned int maxLevel,
														  const unsigned int currentLevel);
		__device__ static Vector3i	CalculateParentVoxId(const Vector3i& voxelPos,
														 const unsigned int maxLevel,
														 const unsigned int currentLevel);
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
inline unsigned int SVODevice::CalculateLevelChildId(const Vector3i& voxelPos,
													 const unsigned int parentLevel,
													 const unsigned int currentLevel)
{
	unsigned int bitSet = 0;
	bitSet |= ((voxelPos[2] >> (currentLevel - parentLevel)) & 0x000000001) << 2;
	bitSet |= ((voxelPos[1] >> (currentLevel - parentLevel)) & 0x000000001) << 1;
	bitSet |= ((voxelPos[0] >> (currentLevel - parentLevel)) & 0x000000001) << 0;
	return bitSet;
}

__device__
inline Vector3i SVODevice::CalculateParentVoxId(const Vector3i& voxelPos,
												const unsigned int parentLevel,
												const unsigned int currentLevel)
{
	assert(currentLevel >= parentLevel);
	Vector3i levelVoxelId;
	levelVoxelId[0] = voxelPos[0] >> (currentLevel - parentLevel);
	levelVoxelId[1] = voxelPos[1] >> (currentLevel - parentLevel);
	levelVoxelId[2] = voxelPos[2] >> (currentLevel - parentLevel);
	return levelVoxelId;
}

__device__ 
inline bool SVODevice::Intersects(HitRecord& record, const RayF& ray) const
{
	Vector3 pos;
	float t, tTotal = 0;
	Vector3 extents = (aabbMax - aabbMin) / static_cast<Vector3>(dimensions - Vector3ui(1u));

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
		Vector3i index = static_cast<Vector3i>((location - aabbMin) / extents);

		// Traverse through tree for that location
		// and find its aabb				
		Node* n = root;
		uint32_t depth = 0;
		while(depth < totalLevel)
		{
			const uint32_t nextNode = n->next;
			if(nextNode == NULL_CHILD) break;
			depth++;

			uint32_t childLoc = CalculateLevelChildId(index, depth, totalLevel);
			n = root + nextNode + childLoc;
		}

		// Generate Traversed AABB
		Vector3f depthExtents = extents * (0x1 << (totalLevel - depth));
		Vector3f remainder = Vector3f(fmodf(location[0], depthExtents[0]),
									  fmodf(location[1], depthExtents[1]),
									  fmodf(location[2], depthExtents[2]));		
		Vector3f localAABBMin = location - remainder;
		Vector3f localAABBMax = localAABBMin + depthExtents;

		// Do an intersection with that level of the AABB
		// This should always enter (by definition)
		if(currentRay.IntersectsAABB(pos, t, localAABBMin, localAABBMax))
		{
			currentRay.AdvanceSelf(t + MathConstants::Epsilon);
			tTotal += (t + MathConstants::Epsilon);
		}
		
		// Chcek if we found a node
		if(n->next == DATA_LEAF)
		{
			// Float index of location
			record.baryCoord = (currentRay.getPosition() - (aabbMin + extents * 0.5f)) / extents;
			record.distance = tTotal;
			
			record.objectId = volumeId;
			record.triangleId = HitRecord::VOLUME_SAMPLE;

			return true;
		}
	}
	return false;
}