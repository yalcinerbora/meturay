#include "ImageMemory.h"

int ImageMemory::PixelFormatToSize(PixelFormat f)
{
    static constexpr int SizeList[static_cast<int>(PixelFormat::END)] =
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

ImageMemory::ImageMemory()
    : segmentSize(Zero3ui)
    , resolution(Zero3ui)
    , segmentOffset(Zero3ui)
    , format(PixelFormat::END)
    , pixelSize(0)
{}

ImageMemory::ImageMemory(const Vector2i& offset,
                         const Vector2i& size,
                         const Vector2i& resolution,
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
    pixelSize = PixelFormatToSize(f);
    Reportion(segmentOffset, segmentSize);
}

void ImageMemory::Reportion(Vector2i start,
                            Vector2i end)
{
    end = Vector2i::Min(resolution, end);
    segmentOffset = start;
    segmentSize = end - start;

    size_t linearSize = segmentSize[0] * segmentSize[1] * PixelFormatToSize(format);
    if(linearSize != 0)
    {
        memory = std::move(DeviceMemory(linearSize));
        Reset();
    }
}

void ImageMemory::Resize(Vector2i res)
{
    assert(segmentSize <= res);
    resolution = res;
}

void ImageMemory::Reset()
{
    size_t pixelCount = static_cast<size_t>(segmentSize[0]) * segmentSize[1];
    if(pixelCount != 0)
        CUDA_CHECK(cudaMemset(memory, 0x0, PixelFormatToSize(format) * pixelCount));
}

Vector2i ImageMemory::SegmentSize() const
{
    return segmentSize;
}

Vector2i ImageMemory::SegmentOffset() const
{
    return segmentOffset;
}

Vector2i ImageMemory::Resolution() const
{
    return resolution;
}

PixelFormat ImageMemory::Format() const
{
    return format;
}

int ImageMemory::PixelSize() const
{
    return pixelSize;
}