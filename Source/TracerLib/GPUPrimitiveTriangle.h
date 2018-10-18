#pragma once
/**

Default Triangle Implementation

Has three types of data
Position, Normal and UV.

All of them should be provided

*/

#include <map>
#include <type_traits>

#include "GPUPrimitiveI.h"
#include "DefaultLeaf.h"
#include "AcceleratorDeviceFunctions.h"

#include "RayLib/DeviceMemory.h"
#include "RayLib/Vector.h"

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
__device__
inline HitResult TriangleClosestHit(// Output
									HitKey& newMat,
									PrimitiveId& newPrimitive,
									TriangleHit& newHit,
									// I-O
									RayReg& rayData,
									// Input
									const TriData& primData,
									const DefaultLeaf& leaf)
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

class GPUPrimitiveTriangle final : public GPUPrimitiveGroupI
{
	public:	
	   	// Type Definitions for kernel generations
		using PrimitiveData						= TriData;
		using HitReg							= TriangleHit;
		static constexpr auto AcceptFunc		= TriangleClosestHit;
		static constexpr auto GenLeafFunc		= GenerateLeaf<PrimitiveData>;
		// 
		static constexpr auto AABBGenFunc		= TriangleClosestHit;
		static constexpr auto AreaGenFunc		= TriangleClosestHit;

	private:
		DeviceMemory							memory;
		PrimitiveData							dData;

		// List of ranges for each batch
		uint64_t								totalPrimitiveCount;
		std::map<uint32_t, Vector2ul>			batchRanges;
	
	public:
		// Constructors & Destructor
												GPUPrimitiveTriangle();
												~GPUPrimitiveTriangle() = default;

		// Interface
		// Pirmitive type is used for delegating scene info to this class
		const std::string&						PrimitiveType() const override;
		// Allocates and Generates Data
		SceneError								InitializeGroup(const std::vector<SceneFileNode>& surfaceDatalNodes, double time) override;
		SceneError								ChangeTime(const std::vector<SceneFileNode>& surfaceDatalNodes, double time) override;

		// Access primitive range from Id			
		Vector2ui								PrimitiveBatchRange(uint32_t surfaceDataId) override;

		// Error check
		// Queries in order to check if this primitive group supports certain primitive data
		// Material may need that data
		bool									CanGenerateData(const std::string& s) const override;
};