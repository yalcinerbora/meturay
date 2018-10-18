#pragma once
/**

Default Sphere Implementation
One of the fundamental functional types.

Has two types of data
Position and radius.

All of them should be provided

*/

#include <map>
#include <type_traits>

#include "GPUPrimitiveI.h"
#include "DefaultLeaf.h"
#include "AcceleratorDeviceFunctions.h"

#include "RayLib/DeviceMemory.h"
#include "RayLib/Vector.h"

// Sphere memory layout
struct SphereData
{
	const Vector4f* centersRadius;
};

// Hit of sphere is spherical coordinates
using SphereHit = Vector2f;

// Sphere Hit Acceptance
__device__ __host__
inline HitResult SphereClosestHit(// Output
								  HitKey& newMat,
								  PrimitiveId& newPrimitive,
								  SphereHit& newHit,
								  // I-O
								  RayReg& rayData,
								  // Input
								  const SphereData& primData,
								  const DefaultLeaf& leaf)
{
	// Get Packed data and unpack
	Vector4f data = primData.centersRadius[leaf.primitiveId];
	Vector3f center = data;
	float radius = data[3];

	// Do Intersecton test	
	Vector3 pos; float newT;
	bool intersects = rayData.ray.IntersectsSphere(pos, newT, center, radius);

	// Check if the hit is closer
	bool closerHit = intersects && (newT < rayData.tMin);
	if(closerHit)
	{
		newMat = leaf.matId;
		newPrimitive = leaf.primitiveId;

		// Gen Spherical Coords (R can be fetched using primitiveId)
		Vector3 relativeCoord = pos - center;
		float tetha = acos(relativeCoord[2] / radius);
		float phi = atan2(relativeCoord[1], relativeCoord[0]);
		newHit = Vector2(tetha, phi);
	}
	return HitResult{false, closerHit};
}

class GPUPrimitiveSphere final : public GPUPrimitiveGroupI
{
	public:	
		// Type Definitions for kernel generations
		using PrimitiveData						= SphereData;
		using HitReg							= SphereHit;
		static constexpr auto AcceptFunc		= SphereClosestHit;
		static constexpr auto GenLeafFunc		= GenerateLeaf<PrimitiveData>;
		// 		
		static constexpr auto AABBGenFunc		= SphereClosestHit;
		static constexpr auto AreaGenFunc		= SphereClosestHit;

	private:
		DeviceMemory							memory;
		PrimitiveData							dData;

		// List of ranges for each batch
		uint64_t								totalPrimitiveCount;
		std::map<uint32_t, Vector2ui>			batchRanges;
	
	public:
		// Constructors & Destructor
												GPUPrimitiveSphere();
												~GPUPrimitiveSphere() = default;

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