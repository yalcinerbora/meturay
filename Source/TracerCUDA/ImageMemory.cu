#include "ImageMemory.h"
#include "ImageFunctions.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

#include "RayLib/MemoryAlignment.h"
#include "RayLib/Log.h"
#include "RayLib/TracerError.h"
#include "ImageIO/ImageIOI.h"

// Template lambda to ease of read

template <class T>
using ImgKernelFunc = void(&)(ImageGMem<T> mem, size_t totalPixelCount);

template <class T, ImgKernelFunc<T> KF>
struct KC
{
    size_t          pixelCount;
    ImageMemory&    mem;
    const CudaGPU&  gpu;

    void operator()() const
    {
        gpu.GridStrideKC_X(0, (cudaStream_t)0,
                           pixelCount,
                           KF,
                           //
                           mem.GMem<T>(),
                           pixelCount);
    }
};

template <class T>
__device__
void ChangeNaNToColor(T& pixel);

template <>
__device__
void ChangeNaNToColor(float& pixel)
{
    if(isnan(pixel))
    {
        pixel = 1.0e30;
    }
}

template <class T>
__device__
void ChangeNaNToColor(T& pixel)
{
    if(pixel.HasNaN())
    {
        // Push bright magenta to visualize NaN
        pixel = Vector4(1.0e30, 0.0, 1.0e30, 1.0f);
    }
}

template <class T>
__global__
void KCCheckPixels(ImageGMem<T> mem, size_t totalPixelCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalPixelCount;
        threadId += (blockDim.x * gridDim.x))
    {
        T pixel = mem.gPixels[threadId];
        ChangeNaNToColor(pixel);
        mem.gPixels[threadId] = pixel;
    }
}

template <class T>
__global__
void KCResetSamples(ImageGMem<T> mem, size_t totalPixelCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalPixelCount;
        threadId += (blockDim.x * gridDim.x))
    {
        mem.gSampleCounts[threadId] = 0;
        mem.gPixels[threadId] = Zero4f;
    }
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
    : format(f)
    , pixelSize(static_cast<int>(ImageIOI::FormatToPixelSize(format)))
    , segmentSize(size)
    , segmentOffset(offset)
    , resolution(resolution)
{}

void ImageMemory::SetPixelFormat(PixelFormat f, const CudaSystem& s)
{
    format = f;
    pixelSize = static_cast<int>(ImageIOI::FormatToPixelSize(f));
    Reportion(segmentOffset, segmentSize, s);
}

void ImageMemory::Reportion(Vector2i start,
                            Vector2i end,
                            const CudaSystem& system)
{
    end = Vector2i::Min(resolution, end);
    segmentOffset = start;
    segmentSize = end - start;

    size_t pixelCount = static_cast<size_t>(segmentSize[0]) * segmentSize[1];
    size_t sizeOfPixels = ImageIOI::FormatToPixelSize(format) * pixelCount;
    sizeOfPixels = Memory::AlignSize(sizeOfPixels);
    size_t sizeOfPixelCounts = sizeof(float) * pixelCount;

    if(pixelCount != 0)
    {
        memory = std::move(DeviceMemory(sizeOfPixels + sizeOfPixelCounts));

        size_t offset = 0;
        std::uint8_t* dMem = static_cast<uint8_t*>(memory);
        dPixels = dMem + offset;
        offset += sizeOfPixels;
        dSampleCounts = reinterpret_cast<float*>(dMem + offset);
        offset += sizeOfPixelCounts;
        assert(offset == (sizeOfPixels + sizeOfPixelCounts));

        Reset(system);
    }
}

void ImageMemory::Resize(Vector2i res)
{
    assert(segmentSize <= res);
    resolution = res;
}

void ImageMemory::Reset(const CudaSystem&)
{
    size_t pixelCount = static_cast<size_t>(segmentSize[0]) * segmentSize[1];
    if(pixelCount != 0)
    {
        size_t totalBytes = ImageIOI::FormatToPixelSize(format) * pixelCount +
                            sizeof(float) * pixelCount;
        CUDA_CHECK(cudaMemset(memory, 0x0, totalBytes));
    }
}

std::vector<Byte> ImageMemory::GetImageToCPU(const CudaSystem& system)
{
    size_t pixelCount = static_cast<size_t>(segmentSize[0]) * segmentSize[1];
    size_t totalBytes = ImageIOI::FormatToPixelSize(format) * pixelCount +
                        sizeof(float) * pixelCount;
    size_t sampleStart = ImageIOI::FormatToPixelSize(format) * pixelCount;

    if(pixelCount != 0)
    {
        // Pixel Count is relatively small single GPU should handle it
        const CudaGPU& gpu =system.BestGPU();

        switch(format)
        {
            case PixelFormat::R_FLOAT: KC<float, KCCheckPixels>{pixelCount, *this, gpu}(); break;
            case PixelFormat::RG_FLOAT: KC<Vector2f, KCCheckPixels>{pixelCount, *this, gpu}(); break;
            case PixelFormat::RGB_FLOAT: KC<Vector3f, KCCheckPixels>{pixelCount, *this, gpu}(); break;
            case PixelFormat::RGBA_FLOAT: KC<Vector4f, KCCheckPixels>{pixelCount, *this, gpu}(); break;
                break;
            default:
                throw TracerException(TracerError::IMEM_UNKNOWN_PIXEL_FORMAT);
        }
    }

    std::vector<Byte> result(totalBytes);
    // Copy Pixels
    CUDA_CHECK(cudaMemcpy(result.data(),
                          dPixels,
                          ImageIOI::FormatToPixelSize(format) * pixelCount,
                          cudaMemcpyDeviceToHost));
    // Copy Sample Counts
    CUDA_CHECK(cudaMemcpy(result.data() + sampleStart,
                          dSampleCounts,
                          sizeof(float) * pixelCount,
                          cudaMemcpyDeviceToHost));

    // Reset the image(all data is transferred)
    Reset(system);
    return std::move(result);
}