#pragma once
/**

Ray Struct that is mandatory for hit acceleration

	Ray has two different layouts
	One is global memory layout which is optimized for memory acess (and minimize padding)
	Second is register layout which is used for cleaner code

*/

#include <vector>
#include "RayLib/Vector.h"
#include "RayLib/Ray.h"

// Global memory Layout for Rays
struct alignas(32) RayGMem
{
	Vector3		pos;
	float		tMin;
	Vector3		dir;
	float		tMax;
};

// GPU register layout for rays
struct RayReg
{
	RayF							ray;
	float						tMin;
	float						tMax;

								RayReg() = default;
	__device__ __host__			RayReg(const RayGMem* mem,
									   unsigned int loc);

	// Save
	__device__ __host__ void		Update(RayGMem* mem,
									   unsigned int loc);
	__device__ __host__ void		UpdateTMax(RayGMem* mem,
										   unsigned int loc);
};

__device__ __host__
inline RayReg::RayReg(const RayGMem* mem,
					  unsigned int loc)
{
	RayGMem rayGMem = mem[loc];
	ray = RayF(rayGMem.dir, rayGMem.pos);
	tMin = rayGMem.tMin;
	tMax = rayGMem.tMax;
}

__device__ __host__
inline void RayReg::Update(RayGMem* mem,
						   unsigned int loc)
{
	RayGMem rayGMem = 
	{
		ray.getPosition(),
		tMin,
		ray.getDirection(),
		tMax
	};
	mem[loc] = rayGMem;
}

__device__ __host__
inline void RayReg::UpdateTMax(RayGMem* mem,
							   unsigned int loc)
{
	mem[loc].tMax = tMax;
}