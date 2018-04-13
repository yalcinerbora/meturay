#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Random.cuh"
#include "RayLib/DeviceMemory.h"
#include "RayLib/Log.h"

__global__ void KRandomNumbers(RandomStackGMem gMemory,
							   uint32_t* randomNumbers,
							   size_t numberCount)
{
	extern __shared__ uint32_t sStates[];

	RandomGPU rand(gMemory.state, sStates);

	for(int i = 0; i < numberCount; i++)
	{
		uint32_t r = rand.Generate();
		printf("%u\n", r);
		randomNumbers[i] = r;
	}
}

TEST(RandomGPU2, Test)
{
	static constexpr size_t numberCount = 25;
	static constexpr size_t numberSize = numberCount * sizeof(int);

	DeviceMemory randomState(sizeof(int));
	DeviceMemory numbers(numberSize);
	CUDA_CHECK(cudaMemset(static_cast<uint32_t*>(randomState), 0xFA, sizeof(int)));

	const uint32_t* h_data = static_cast<uint32_t*>(numbers);

	// Kernel Call
	KRandomNumbers<<<1, 1, sizeof(uint32_t)>>>({static_cast<uint32_t*>(randomState)},
												static_cast<uint32_t*>(numbers),
												numberCount);
	CUDA_KERNEL_CHECK();
	CUDA_CHECK(cudaDeviceSynchronize());

	for(int i = 0; i < numberCount; i++)
	{
		METU_LOG("%u", h_data[i]);

	}

	METU_LOG("end");
}