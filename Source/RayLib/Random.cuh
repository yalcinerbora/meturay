#pragma once

/**

GPU Linear Congruential Generator

Implementation of Warp Std Generator

*/

#include <limits>
#include <cuda_runtime.h>
#include "CudaConstants.h"
#include "Vector.h"
#include "RandomStructs.h"

#define GLOBAL_ID_X (threadIdx.x + blockDim.x * blockIdx.x)
#define GLOBAL_ID_Y (threadIdx.y + blockDim.y * blockIdx.y)
#define LINEAR_GLOBAL_ID (GLOBAL_ID_X + GLOBAL_ID_Y * gridDim.x)
#define LINEAR_THREAD_ID (threadIdx.x + threadIdx.y * blockDim.x)

class RandomGPU
{
	public:
		static constexpr uint32_t	Max = std::numeric_limits<uint32_t>::max();
		static constexpr uint32_t	Min = std::numeric_limits<uint32_t>::min();
		static constexpr uint32_t	Range = Max-Min;

	private:		 
		 uint32_t*					gStates;
		 uint32_t*					sStates;
		 const Vector3ui			regs;		 
	protected:
	public:
		// Constructor
		__device__					RandomGPU(uint32_t* gStates, uint32_t* sStates);
									RandomGPU(const RandomGPU&) = delete;
		RandomGPU&					operator=(const RandomGPU&) = delete;
		__device__					~RandomGPU();

		// Fundemental Generation Function
		__device__  uint32_t		Generate();
};

static constexpr uint32_t WarpStandard_N = 1024;
static constexpr uint32_t WarpStandard_W = 32;
static constexpr uint32_t WarpStandard_G = 16;
static constexpr uint32_t WarpStandard_SR = 0;
static constexpr uint32_t WarpStandard_Z0 = 2;

__device__ static const uint32_t WarpStandard_Z1[32] =
{
	0,1,0,1,1,1,0,0,1,0,0,1,0,0,1,0,
	0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1
};

__device__ static const uint32_t WarpStandard_Q[2][32] =
{
	{
		29,24,5,23,14,26,11,31,9,3,1,28,0,2,22,20,
		18,15,27,13,10,16,8,17,25,12,19,30,7,6,4,21
	},
	{
		5,14,28,24,19,13,0,17,11,20,7,10,6,15,2,9,8,
		23,4,30,12,25,3,21,26,27,31,18,22,16,29,1
	}
};

__device__
inline RandomGPU::RandomGPU(uint32_t* gStates, uint32_t* sStates)
	: gStates(gStates)
	, sStates(sStates)
	, regs(													  WarpStandard_Z1[LINEAR_THREAD_ID % warpSize],
		   LINEAR_THREAD_ID - (LINEAR_THREAD_ID % warpSize) + WarpStandard_Q[0][LINEAR_THREAD_ID % warpSize],
		   LINEAR_THREAD_ID - (LINEAR_THREAD_ID % warpSize) + WarpStandard_Q[1][LINEAR_THREAD_ID % warpSize])
{
	unsigned int stateOff = LINEAR_GLOBAL_ID;
	sStates[LINEAR_THREAD_ID] = gStates[stateOff];
}

__device__
inline uint32_t RandomGPU::Generate()
{
	unsigned int t0 = sStates[regs[1]];
	unsigned int t1 = sStates[regs[2]];
	unsigned int res = (t0 << WarpStandard_Z0) ^ (t1 >> regs[0]);

	__syncthreads();
	sStates[LINEAR_THREAD_ID] = res;
	__syncthreads();

	return t0 + t1;
}

__device__
inline RandomGPU::~RandomGPU()
{
	unsigned int stateOff = LINEAR_GLOBAL_ID;
	gStates[stateOff] = sStates[LINEAR_THREAD_ID];
}

// Pseduo Uniform Generation
__device__ inline float RandFloat01(RandomGPU& r)
{
	return static_cast<float>(RandomGPU::Min) +
			static_cast<float>(r.Generate()) / static_cast<float>(RandomGPU::Range);
}

__device__ inline float RandMMFloat(RandomGPU& r, const Vector2& minMax)
{
	float f = RandFloat01(r);
	float range = minMax[1] - minMax[0];
	return minMax[0] + f / range;
}

__device__ inline float RandMMInt(RandomGPU& r, const Vector2i minMax)
{
	float f = RandMMFloat(r, Vector2(static_cast<float>(minMax[0]),
								   static_cast<float>(minMax[1])));
	float range = static_cast<float>(minMax[1] - minMax[0]);
	return minMax[0] + static_cast<int>(roundf(f / range));
}

__device__ inline float RandMMUInt(RandomGPU& r, const Vector2ui minMax)
{
	float f = RandMMFloat(r, Vector2(static_cast<float>(minMax[0]),
									 static_cast<float>(minMax[1])));
	float range = static_cast<float>(minMax[1] - minMax[0]);
	return minMax[0] + static_cast<uint32_t>(roundf(f / range));
}