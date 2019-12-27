#pragma once

#include "ImageStructs.h"
#include <numeric>

static constexpr uint32_t MAX_UINT32 = std::numeric_limits<uint32_t>::max();

template<class T>
__device__
void ImageAccumulatePixel(ImageGMem<T>&, uint32_t location, const T& data);

template<class T>
__device__
void ImageAverageSample(ImageGMem<T>&, uint32_t location);

template<>
__device__
inline void ImageAccumulatePixel(ImageGMem<Vector4f>& img, uint32_t location, const Vector4f& data)
{
	img.gPixels[location] = data;
	img.gSampleCount[location] = 1;

	//// Inc Sample
	//atomicInc(&img.gSampleCount[location], MAX_UINT32);

	//// Add value
	//atomicAdd(&img.gPixels[location][0], data[0]);
	//atomicAdd(&img.gPixels[location][1], data[1]);
	//atomicAdd(&img.gPixels[location][2], data[2]);
	//atomicAdd(&img.gPixels[location][3], data[3]);
}

template<>
__device__
inline void ImageAccumulatePixel(ImageGMem<Vector3f>& img, uint32_t location, const Vector3f& data)
{
	// Inc Sample
	atomicInc(&img.gSampleCount[location], MAX_UINT32);

	// Add value
	atomicAdd(&img.gPixels[location][0], data[0]);
	atomicAdd(&img.gPixels[location][1], data[1]);
	atomicAdd(&img.gPixels[location][2], data[2]);
}

template<>
__device__
inline void ImageAverageSample<Vector4f>(ImageGMem<Vector4f>& img, uint32_t location)
{
	const float sampleCount = static_cast<float>(img.gSampleCount[location]);
	if(sampleCount != 0)
	{
		img.gPixels[location][0] /= sampleCount;
		img.gPixels[location][1] /= sampleCount;
		img.gPixels[location][2] /= sampleCount;
		img.gPixels[location][3] /= sampleCount;
	}
}

template<>
__device__
inline void ImageAverageSample<Vector3f>(ImageGMem<Vector3f>& img, uint32_t location)
{
	const float sampleCount = static_cast<float>(img.gSampleCount[location]);
	if(sampleCount != 0)
	{
		img.gPixels[location][0] /= sampleCount;
		img.gPixels[location][1] /= sampleCount;
		img.gPixels[location][2] /= sampleCount;
	}
}