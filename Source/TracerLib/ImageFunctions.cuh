#pragma once

#include "ImageStructs.h"
#include <numeric>

static constexpr uint32_t MAX_UINT32 = std::numeric_limits<uint32_t>::max();

template<class T>
__device__
void ImageAccumulatePixel(ImageGMem<T>&, uint32_t location, const T& data);

template<class T>
__device__
void ImageSetSample(ImageGMem<T>&, uint32_t location, uint32_t sampleCount);

template<class T>
__device__
void ImageAddSample(ImageGMem<T>&, uint32_t location, uint32_t sampleCount);

template<>
__device__
inline void ImageAccumulatePixel(ImageGMem<Vector4f>& img, uint32_t location, const Vector4f& data)
{
	// Add value
	atomicAdd(&img.gPixels[location][0], data[0]);
	atomicAdd(&img.gPixels[location][1], data[1]);
	atomicAdd(&img.gPixels[location][2], data[2]);
	atomicAdd(&img.gPixels[location][3], data[3]);
}

template<>
__device__
inline void ImageAccumulatePixel(ImageGMem<Vector3f>& img, uint32_t location, const Vector3f& data)
{
	// Add value
	atomicAdd(&img.gPixels[location][0], data[0]);
	atomicAdd(&img.gPixels[location][1], data[1]);
	atomicAdd(&img.gPixels[location][2], data[2]);
}

template<class T>
__device__
inline void ImageSetSample<Vector4f>(ImageGMem<T>& img, uint32_t location, uint32_t sampleCount)
{
	img.gSampleCounts[location] = sampleCount;
}

template<class T>
__device__
inline void ImageAddSample<Vector4f>(ImageGMem<T>& img, uint32_t location, uint32_t sampleCount)
{
	atomicAdd(&img.gSampleCounts[location], sampleCount);
}