#pragma once
/**

Some default implementations for triangle primitive

Triangle primitive is packed as struct of array

each classification has a tag wich represents

P: position
N: normal
U: uv



*/

#include <map>
#include <type_traits>

#include "GPUPrimitiveI.h"
#include "HitStructs.cuh"
#include "AcceleratorDeviceFunctions.h"

#include "RayLib/DeviceMemory.h"
#include "RayLib/Vector.h"

// Triangle Data Layouts
struct TriData
{
	Vector4f* positionsU;
	Vector4f* normalsV;
};

// Triangle Hits
using TriangleHit = Vector2f;

// Triangle Leaf Structs
struct Leaf
{
	PrimitiveId		primitiveId;
	HitKey			matId;
};

// Triangle Hit Acceptance
__device__ 
HitResult TriangleClosestHit(// Output
							 HitKey& newMat,
							 PrimitiveId& newPrimitive,
							 TriangleHit& newHit,
							 // I-O
							 RayReg& rayData,
							 // Input
							 const TriData& primData,
							 const Leaf& leaf)
{
	// Get Position
	Vector3 position0 = primData.positionsU[leaf.primitiveId * 3 + 0];
	Vector3 position1 = primData.positionsU[leaf.primitiveId * 3 + 1];
	Vector3 position2 = primData.positionsU[leaf.primitiveId * 3 + 2];

	// Do Intersecton test
	float newT;
	Vector3 baryCoords;
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
		struct TriData
		{
			Vector4f*							positionsU;
			Vector4f*							normalsV;
		};

		// Type Definitions for kernel generations
		using PrimitiveData						= TriData;
		using HitReg							= Vector2f; // Barycentric coords
		static constexpr auto AcceptFunc		= TriangleClosestHit;
		// 
		static constexpr auto AABBGenFunc		= TriangleClosestHit;
		static constexpr auto AreaGenFunc		= TriangleClosestHit;

	private:
		DeviceMemory							memory;
		PrimitiveData							data;

		// List of ranges for each batch
		std::map<uint32_t, Vector2ui>			batchRanges;
	
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