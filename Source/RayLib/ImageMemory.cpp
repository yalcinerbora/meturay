#include "ImageMemory.h"

size_t ImageMemory::PixelFormatToSize(PixelFormat f)
{
	static constexpr size_t SizeList[] =
	{
		1,
		2,
		3,
		4,

		2,
		4,
		6,
		8,

		2,
		4,
		6,
		8,

		4,
		8,
		12,
		16
	};
	return SizeList[static_cast<int>(f)];
}

ImageMemory::ImageMemory(const Vector2ui& offset,
						 const Vector2ui& size,
						 const Vector2ui& resolution,
						 PixelFormat f)
	: segmentSize(size)
	, segmentOffset(offset)
	, resolution(resolution)
	, format(f)
	, pixelSize(PixelFormatToSize(format))
{}

void ImageMemory::SetPixelFormat(PixelFormat f)
{
	format = f;
	Reportion(segmentOffset, segmentSize);
}

void ImageMemory::Reportion(const Vector2ui& offset,
							const Vector2ui& size)
{
	segmentOffset = offset;
	segmentSize = size;

	size_t linearSize = size[0] * size[1] * PixelFormatToSize(format);

	if(linearSize != 0)
	{
		memory = std::move(DeviceMemory(linearSize));
		Reset();
	}
}

void ImageMemory::Resize(const Vector2ui& res)
{
	assert(segmentSize <= res);
	resolution = res;
}

void ImageMemory::Reset()
{
	CUDA_CHECK(cudaDeviceSynchronize());
	size_t pixelCount = segmentSize[0] * segmentSize[1];
	if(pixelCount != 0)
		CUDA_CHECK(cudaMemset(memory, 0x0, PixelFormatToSize(format) * pixelCount));
}

Vector2ui ImageMemory::SegmentSize() const
{
	return segmentSize;
}

Vector2ui ImageMemory::SegmentOffset() const
{
	return segmentOffset;
}

Vector2ui ImageMemory::Resolution() const
{
	return resolution;
}