#pragma once

/**

GPU Linear Congruential Generator

Implementation of Warp Std Generator

*/

#include <cuda_runtime.h>
#include "CudaConstants.h"
#include "Vector.h"

struct RandomStackGMem
{
	uint32_t* state;
};

class RandomGPU
{
	private:
		 static constexpr uint32_t WarpStandard_N = 1024;
		 static constexpr uint32_t WarpStandard_W = 32;
		 static constexpr uint32_t WarpStandard_G = 16;
		 static constexpr uint32_t WarpStandard_SR = 0;
		 static constexpr uint32_t WarpStandard_Z0 = 2;

		 uint32_t*				gStates;
		 uint32_t*				sStates;
		 const Vector3ui		regs;

	protected:
	public:
		// Constructor
		__device__				RandomGPU(uint32_t* gStates, uint32_t* sStates);
								RandomGPU(const RandomGPU&) = delete;
		RandomGPU&				operator=(const RandomGPU&) = delete;
		__device__				~RandomGPU();

		// Fundemental Generation Function
		__device__  uint32_t	Generate();
};

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
	, regs(											WarpStandard_Z1[threadIdx.x % warpSize],
		   threadIdx.x - (threadIdx.x % warpSize) + WarpStandard_Q[0][threadIdx.x % warpSize],
		   threadIdx.x - (threadIdx.x % warpSize) + WarpStandard_Q[1][threadIdx.x % warpSize])
{
	unsigned int stateOff = blockDim.x * blockIdx.x + threadIdx.x;
	sStates[threadIdx.x] = gStates[stateOff];
}

__device__
inline uint32_t RandomGPU::Generate()
{
	unsigned int t0 = sStates[regs[1]];
	unsigned int t1 = sStates[regs[2]];
	unsigned int res = (t0 << WarpStandard_Z0) ^ (t1 >> regs[0]);

	__syncthreads();
	sStates[threadIdx.x] = res;
	__syncthreads();

	return t0 + t1;
}

__device__
inline RandomGPU::~RandomGPU()
{
	unsigned int stateOff = blockDim.x * blockIdx.x * 1 + threadIdx.x * 1;
	gStates[stateOff] = sStates[threadIdx.x];
}