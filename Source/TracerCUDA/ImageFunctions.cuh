#pragma once

#include "ImageStructs.h"
#include <numeric>

static constexpr uint32_t MAX_UINT32 = std::numeric_limits<uint32_t>::max();

template<class T>
__device__ inline
void AccumulateRaySample(CamSampleGMem<T>&, uint32_t location, const T& data);

template<>
__device__ inline
void AccumulateRaySample(CamSampleGMem<Vector4f>& samples, uint32_t location, const Vector4f& data)
{
	// Add value
	atomicAdd(&samples.gValues[location][0], data[0]);
	atomicAdd(&samples.gValues[location][1], data[1]);
	atomicAdd(&samples.gValues[location][2], data[2]);
	atomicAdd(&samples.gValues[location][3], data[3]);
}

template<>
__device__ inline
void AccumulateRaySample(CamSampleGMem<Vector3f>& samples, uint32_t location, const Vector3f& data)
{
	// Add value
	atomicAdd(&samples.gValues[location][0], data[0]);
	atomicAdd(&samples.gValues[location][1], data[1]);
	atomicAdd(&samples.gValues[location][2], data[2]);
}

template<>
__device__ inline
void AccumulateRaySample(CamSampleGMem<float>& samples, uint32_t location, const float& data)
{
	// Add value
	atomicAdd(&samples.gValues[location], data);
}