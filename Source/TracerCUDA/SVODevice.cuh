#pragma once
/**

Sparse Voxel Octree Implementation for GPU's using CUDA.

Basic SVO implementation used to accelerate GPUVolume intersection
and traversal.


*/

#include <cstdint>

//#include "RayLib/RayHitStructs.h"
#include "RayLib/DeviceMemory.h"
#include "RayLib/Vector.h"

struct VolumeDeviceData;
struct SVODeviceData;
struct HitRecord;
class VolumeI;

struct SVODeviceData
{
	struct alignas(sizeof(uint32_t)) Node
	{
		uint32_t					next;
	};

	static constexpr uint32_t		DATA_LEAF = 0xFFFFFFFD;
	static constexpr uint32_t		ALLOCATION_IN_PROGRESS = 0xFFFFFFFE;
	static constexpr uint32_t		NULL_CHILD = 0xFFFFFFFF;

	// Extents
	Vector3							aabbMin;
	Vector3							aabbMax;

	Vector3ui						dimensions;
	uint32_t						totalLevel;

	uint32_t						volumeId;

	Node*							d_root;

	// Intersection
	//__device__ bool					Intersects(HitRecord&, const RayF&) const;

	// Helper Functions
	__device__ static uint32_t		CalculateLevelChildId(const Vector3i& voxelPos,
															const unsigned int maxLevel,
															const unsigned int currentLevel);
	__device__ static Vector3i		CalculateParentVoxId(const Vector3i& voxelPos,
															const unsigned int maxLevel,
															const unsigned int currentLevel);
	__device__ uint32_t				AtomicAllocateNode(Node* gNode, uint32_t* gLevelAllocator);
};

class SVODevice : public SVODeviceData
{
	private:			
		DeviceMemory					mem;
		
		uint32_t*						d_allocator;
		uint32_t						totalNodeCount;

		static constexpr size_t			InitialNodeCount = 256 * 256 * 256 * 4;
		static constexpr float			NodeIncrementRatio = 2.0f;

		//
		void							IncreaseMemory(uint32_t nodeCount);

	public:
		// Constructors & Destructor
										SVODevice();
										SVODevice(const SVODevice&) = delete;
										SVODevice(SVODevice&&) = default;
		SVODevice&						operator=(const SVODevice&) = delete;
		SVODevice&						operator=(SVODevice&&) = default;
										~SVODevice() = default;

		// Construction & Allocation
		void							ConstructDevice(const Vector3ui& volumeSize, 
														const VolumeI&);

		// Misc
		size_t							TotalNodeSize();
};

__device__
inline unsigned int SVODeviceData::CalculateLevelChildId(const Vector3i& voxelPos,
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
inline Vector3i SVODeviceData::CalculateParentVoxId(const Vector3i& voxelPos,
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

//__device__ 
//inline bool SVODeviceData::Intersects(HitRecord& record, const RayF& ray) const
//{
//	Vector3 pos;
//	float t, tTotal = 0;
//	Vector3 extents = (aabbMax - aabbMin) / static_cast<Vector3>(dimensions - Vector3ui(1u));
//
//	// Initially check if ray is outside of the box
//	// And advance it to the nearest edge + epsilon
//	RayF currentRay = ray;
//	if(currentRay.IntersectsAABB(pos, t, aabbMin, aabbMax) &&
//	   pos[0] == aabbMin[0] ||
//	   pos[1] == aabbMin[1] || 
//	   pos[2] == aabbMin[2])
//	{
//		currentRay.AdvanceSelf(t + MathConstants::Epsilon);
//		tTotal += (t + MathConstants::Epsilon);
//	}
//
//	// Traverse through volume until ray is outside
//	// Or code will break out if it hits
//	while(currentRay.IntersectsAABB(pos, t, aabbMin, aabbMax))
//	{
//		// Find relative index of position
//		Vector3f location = ray.getPosition();
//		Vector3i index = static_cast<Vector3i>((location - aabbMin) / extents);
//
//		// Traverse through tree for that location
//		// and find its aabb				
//		Node* n = d_root;
//		uint32_t depth = 0;
//		while(depth < totalLevel)
//		{
//			const uint32_t nextNode = n->next;
//			if(nextNode == NULL_CHILD) break;
//			depth++;
//
//			uint32_t childLoc = CalculateLevelChildId(index, depth, totalLevel);
//			n = d_root + nextNode + childLoc;
//		}
//
//		// Generate Traversed AABB
//		Vector3f depthExtents = extents * static_cast<float>(0x1 << (totalLevel - depth));
//		Vector3f remainder = Vector3f(fmodf(location[0], depthExtents[0]),
//									  fmodf(location[1], depthExtents[1]),
//									  fmodf(location[2], depthExtents[2]));		
//		Vector3f localAABBMin = location - remainder;
//		Vector3f localAABBMax = localAABBMin + depthExtents;
//
//		// Do an intersection with that level of the AABB
//		// This should always enter (by definition)
//		if(currentRay.IntersectsAABB(pos, t, localAABBMin, localAABBMax))
//		{
//			currentRay.AdvanceSelf(t + MathConstants::Epsilon);
//			tTotal += (t + MathConstants::Epsilon);
//		}
//		
//		// Chcek if we found a node
//		if(n->next == DATA_LEAF)
//		{
//			// Float index of location
//			record.baryCoord = (currentRay.getPosition() - (aabbMin + extents * 0.5f)) / extents;
//			record.distance = tTotal;
//			
//			record.objectId = volumeId;
//			record.triangleId = HitRecord::VOLUME_SAMPLE;
//
//			return true;
//		}
//	}
//	return false;
//}