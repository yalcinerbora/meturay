#pragma once
/**

Mandatory structures that are used
in tracing. Memory optimized for GPU batch fetch

*/

#include <vector>
#include "Vector.h"
#include "Ray.h"

#include <thrust/sort.h>

struct alignas(16) Vec3AndUInt
{
	Vector3				vec;
	unsigned int		uint;
};

struct ConstRayRecordGMem
{
	const Vector4*		posAndMedium;
	const Vec3AndUInt*	dirAndPixId;
	const Vec3AndUInt*	radAndSampId;
};

struct RayRecordGMem
{
	Vector4*			posAndMedium;
	Vec3AndUInt*		dirAndPixId;
	Vec3AndUInt*		radAndSampId;

	constexpr operator	ConstRayRecordGMem() const;
};

struct RayRecordCPU
{
	std::vector<Vector4>		posAndMedium;
	std::vector<Vec3AndUInt>	dirAndPixId;
	std::vector<Vec3AndUInt>	radAndSampId;
};

// RayRecord struct is allocated inside thread (GPU register)
struct RayRecord
{
	RayF					ray;
	float					medium;
	unsigned int			pixelId;
	unsigned int			sampleId;
	Vector3					totalRadiance;

							RayRecord() = default;
	__device__ __host__		RayRecord(const ConstRayRecordGMem& mem,
									  unsigned int loc);
	__device__ __host__		RayRecord(const RayRecordGMem& mem,
									  unsigned int loc);
};

struct ConstHitRecordGMem
{
	const Vec3AndUInt*	baryAndObjId;
	const unsigned int*	triId;
	const float*		distance;
};

struct HitRecordGMem
{
	Vec3AndUInt*		baryAndObjId;
	unsigned int*		triId;
	float*				distance;

	constexpr operator	ConstHitRecordGMem() const;
};

struct HitRecordCPU
{
	std::vector<Vec3AndUInt>	baryAndObjId;
	std::vector<unsigned int>	triId;
	std::vector<float>			distance;
};

// HitRecord struct is allocated inside thread (GPU register)
struct HitRecord
{
	Vector3					baryCoord;
	int						objectId;
	int						triangleId;
	float					distance;

	// Constructor & Destrctor
							HitRecord() = default;
	__device__ __host__		HitRecord(const HitRecordGMem& mem, 
									  unsigned int loc);
	__device__ __host__		HitRecord(const ConstHitRecordGMem& mem,
									  unsigned int loc);
};

// Implementations
constexpr RayRecordGMem::operator ConstRayRecordGMem() const
{
	return 
	{
		posAndMedium,
		dirAndPixId,
		radAndSampId
	};
}

constexpr HitRecordGMem::operator ConstHitRecordGMem() const
{
	return
	{
		baryAndObjId,
		triId,
		distance
	};
}

__device__ __host__
inline RayRecord::RayRecord(const RayRecordGMem& mem, 
							unsigned int loc)
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

__device__ __host__
inline RayRecord::RayRecord(const ConstRayRecordGMem& mem,
							unsigned int loc)
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

__device__ __host__
inline HitRecord::HitRecord(const HitRecordGMem& mem, unsigned int loc)
{
	// Load coalesced
	Vec3AndUInt baryObj = mem.baryAndObjId[loc];
	triangleId = mem.triId[loc];
	distance = mem.distance[loc];

	// Assign in register memory of the Multiprocessor
	baryCoord = baryObj.vec;
	objectId = baryObj.uint;
	
}

__device__ __host__
inline HitRecord::HitRecord(const ConstHitRecordGMem& mem,
							unsigned int loc)
{
	// Load coalesced
	Vec3AndUInt baryObj = mem.baryAndObjId[loc];
	triangleId = mem.triId[loc];
	distance = mem.distance[loc];

	// Assign in register memory of the Multiprocessor
	baryCoord = baryObj.vec;
	objectId = baryObj.uint;
}