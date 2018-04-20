#pragma once
/**

Mandatory structures that are used
in tracing. Memory optimized for GPU batch fetch

*/

#include "Vector.h"
#include "Ray.h"



struct alignas(16) Vec3AndUInt
{
	Vector3				vec;
	unsigned int		uint;
};

struct ConstRayStackGMem
{
	const Vector4*		posAndMedium;
	const Vec3AndUInt*	dirAndPixId;
	const Vec3AndUInt*	radAndSampId;
};

struct RayStackGMem
{
	Vector4*			posAndMedium;
	Vec3AndUInt*		dirAndPixId;
	Vec3AndUInt*		radAndSampId;

	constexpr operator	ConstRayStackGMem();
};

// Ray Stack should not be used on GMem since its not optimized
struct RayStack
{
	RayF			ray;
	float			medium;
	unsigned int	pixelId;
	unsigned int	sampleId;
	Vector3			totalRadiance;

					RayStack() = default;
					RayStack(const RayStackGMem& mem, unsigned int loc);
};

struct HitRecordGMem
{
	Vec3AndUInt*	baryAndObjId;
	unsigned int*	triId;

	constexpr operator ConstHitRecordGMem();
};

struct ConstHitRecordGMem
{
	const Vec3AndUInt*	baryAndObjId;
	const unsigned int*	triId;
};

struct RayRecodCPU
{
	HitRecordGMem cpuHits;
	RayStackGMem cpuRays;
};

struct ConstRayRecodCPU
{
	ConstHitRecordGMem cpuHits;
	ConstRayStackGMem cpuRays;
};

// Hit record should not be used on GMem since its not optimized
struct HitRecord
{
	Vector3			baryCoord;
	int				objectId;
	int				triangleId;

	// Constructor & Destrctor
					HitRecord() = default;
					HitRecord(const HitRecordGMem& mem, unsigned int loc);
};

// Implementations
constexpr RayStackGMem::operator ConstRayStackGMem()
{
	return 
	{
		posAndMedium,
		dirAndPixId,
		radAndSampId
	};
}

constexpr HitRecordGMem::operator ConstHitRecordGMem()
{
	return
	{
		baryAndObjId,
		triId
	};
}

inline RayStack::RayStack(const RayStackGMem& mem, unsigned int loc)
	: ray(Zero3, Zero3)
{
	// Load coalesced
	Vector4 posMed = mem.posAndMedium[loc];
	Vec3AndUInt dirPix = mem.dirAndPixId[loc];
	Vec3AndUInt radSam = mem.radAndSampId[loc];

	// Assign in register memory of the Multiprocessor
	ray = RayF(dirPix.vec, Vector3(posMed[0], posMed[1], posMed[2]));
	medium = posMed[3];
	pixelId = dirPix.uint;
	sampleId = radSam.uint;
	totalRadiance = radSam.vec;
}

inline HitRecord::HitRecord(const HitRecordGMem& mem, unsigned int loc)
{
	// Load coalesced
	Vec3AndUInt baryObj = mem.baryAndObjId[loc];
	triangleId = mem.triId[loc];

	// Assign in register memory of the Multiprocessor
	baryCoord = baryObj.vec;
	objectId = baryObj.uint;
}