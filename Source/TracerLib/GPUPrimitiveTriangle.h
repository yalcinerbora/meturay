#pragma once
/**

Default Triangle Implementation

Has three types of data
Position, Normal and UV.

All of them should be provided

*/

#include <map>
#include <type_traits>

#include "RayLib/Vector.h"
#include "RayLib/Triangle.h"

#include "DefaultLeaf.h"
#include "GPUPrimitiveP.cuh"
#include "DeviceMemory.h"

// Triangle Memory Layout
struct TriData
{
	const Vector4f* positionsU;
	const Vector4f* normalsV;
};

// Triangle Hit is barycentric coordinates
// c is (1-a-b) thus it is not stored.
using TriangleHit = Vector2f;

// Triangle Hit Acceptance
__device__ __host__
inline HitResult TriangleClosestHit(// Output
									HitKey& newMat,
									PrimitiveId& newPrimitive,
									TriangleHit& newHit,
									// I-O
									RayReg& rayData,
									// Input									
									const DefaultLeaf& leaf,
									const TriData& primData)
{
	// Get Position
	Vector3 position0 = primData.positionsU[leaf.primitiveId * 3 + 0];
	Vector3 position1 = primData.positionsU[leaf.primitiveId * 3 + 1];
	Vector3 position2 = primData.positionsU[leaf.primitiveId * 3 + 2];

	// Do Intersecton test	
	Vector3 baryCoords; float newT;
	bool intersects = rayData.ray.IntersectsTriangle(baryCoords, newT,
													 position0, position1, position2,
													 false);

	// Check if the hit is closer
	bool closerHit = intersects && (newT < rayData.tMin);
	if(closerHit)
	{
		newMat = leaf.matId;
		newPrimitive = leaf.primitiveId;
		newHit = Vector2(baryCoords[0], baryCoords[1]);
	}
	return HitResult{false, closerHit};
}


__device__ __host__
inline AABB3f GenerateAABBTriangle(PrimitiveId primitiveId, const TriData& primData)
{
	// Get Position
	Vector3 position0 = primData.positionsU[primitiveId * 3 + 0];
	Vector3 position1 = primData.positionsU[primitiveId * 3 + 1];
	Vector3 position2 = primData.positionsU[primitiveId * 3 + 2];
	
	return Triangle::BoundingBox(position0, position1, position2);
}

__device__ __host__
inline float GenerateAreaTriangle(PrimitiveId primitiveId, const TriData& primData)
{
	// Get Position
	Vector3 position0 = primData.positionsU[primitiveId * 3 + 0];
	Vector3 position1 = primData.positionsU[primitiveId * 3 + 1];
	Vector3 position2 = primData.positionsU[primitiveId * 3 + 2];
	// CCW
	Vector3 vec0 = position1 - position0;
	Vector3 vec1 = position2 - position0;

	return Cross(vec0, vec1).Length() * 0.5f;
}

class GPUPrimitiveTriangle final
	: public GPUPrimitiveGroup<TriangleHit, TriData, DefaultLeaf,
							   TriangleClosestHit, GenerateDefaultLeaf,
							   GenerateAABBTriangle, GenerateAreaTriangle>
{
	public:
		static constexpr const char*			TypeName = "Triangle";

	private:		
		DeviceMemory							memory;

		// List of ranges for each batch
		uint64_t								totalPrimitiveCount;
		std::map<uint32_t, Vector2ul>			batchRanges;
		std::map<uint32_t, AABB3>				batchAABBs;

	protected:
	public:
		// Constructors & Destructor
												GPUPrimitiveTriangle();
												~GPUPrimitiveTriangle() = default;

		// Interface
		// Pirmitive type is used for delegating scene info to this class
		const char*								Type() const override;
		// Allocates and Generates Data
		SceneError								InitializeGroup(const std::set<SceneFileNode>& surfaceDatalNodes, double time) override;
		SceneError								ChangeTime(const std::set<SceneFileNode>& surfaceDatalNodes, double time) override;

		// Access primitive range from Id			
		Vector2ul								PrimitiveBatchRange(uint32_t surfaceDataId) const override;
		AABB3									PrimitiveBatchAABB(uint32_t surfaceDataId) const override;

		// Error check
		// Queries in order to check if this primitive group supports certain primitive data
		// Material may need that data
		bool									CanGenerateData(const std::string& s) const override;
};