#include "ImageMemory.h"
#include "ImageFunctions.cuh"
#include "CudaConstants.h"

__global__ void KCAverageSamples(ImageGMem<Vector4f> mem, size_t totalPixelCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalPixelCount;
        threadId += (blockDim.x * gridDim.x))
    {
        ImageAverageSample(mem, threadId);
    }
}

__global__ void KCResetSamples(ImageGMem<Vector4f> mem, size_t totalPixelCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalPixelCount;
        threadId += (blockDim.x * gridDim.x))
    {
        mem.gSampleCount[threadId] = 0;
        mem.gPixels[threadId][0] = 0.0f;
        mem.gPixels[threadId][1] = 0.0f;
        mem.gPixels[threadId][2] = 0.0f;
    }
}

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

void ImageMemory::SetPixelFormat(PixelFormat f, const CudaSystem& s)
{
    format = f;
    pixelSize = PixelFormatToSize(f);
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
    size_t sizeOfPixels = PixelFormatToSize(format) * pixelCount;
    sizeOfPixels = AlignByteCount * ((sizeOfPixels + (AlignByteCount - 1)) / AlignByteCount);
    size_t sizeOfPixelCounts = sizeof(uint32_t) * pixelCount;

    if(pixelCount != 0)
    {
        memory = std::move(DeviceMemory(sizeOfPixels + sizeOfPixelCounts));
        
        size_t offset = 0;
        std::uint8_t* dMem = static_cast<uint8_t*>(memory);
        dPixels = dMem + offset;
        offset += sizeOfPixels;
        dSampleCounts = reinterpret_cast<uint32_t*>(dMem + offset);
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

#include "TracerDebug.h"

void ImageMemory::Reset(const CudaSystem& system)
{
    size_t pixelCount = static_cast<size_t>(segmentSize[0]) * segmentSize[1];
    if(pixelCount != 0)
    {
        //// Pixel Count is relatively small single GPU should handle it
        //const CudaGPU& gpu = *(system.GPUList().begin());
        //// TODO: Do generic image handling
        //gpu.GridStrideKC_X(0, (cudaStream_t)0,
        //                   pixelCount,
        //                   KCResetSamples,
        //                   //
        //                   GMem<Vector4f>(),
        //                   pixelCount);

        size_t totalBytes = PixelFormatToSize(format) * pixelCount +
                            sizeof(uint32_t) * pixelCount;
        CUDA_CHECK(cudaMemset(memory, 0x0, totalBytes));

    }

}

std::vector<Byte> ImageMemory::GetImageToCPU(const CudaSystem& system)
{
    size_t pixelCount = static_cast<size_t>(segmentSize[0]) * segmentSize[1];
    size_t totalBytes = PixelFormatToSize(format) * pixelCount +
                        sizeof(uint32_t) * pixelCount;
    size_t sampleStart = PixelFormatToSize(format) * pixelCount;

    if(pixelCount != 0)
    {
    // Pixel Count is relatively small single GPU should handle it
        const CudaGPU& gpu = *(system.GPUList().begin());

        gpu.GridStrideKC_X(0, (cudaStream_t)0,
                           pixelCount,
                           KCAverageSamples,
                           //
                           GMem<Vector4f>(),
                           pixelCount);
    }

    std::vector<Byte> result(totalBytes);
    // Copy Pixels
    CUDA_CHECK(cudaMemcpy(result.data(),
                          dPixels,
                          PixelFormatToSize(format) * pixelCount,
                          cudaMemcpyDeviceToHost));
    // Copy Sample Counts
    CUDA_CHECK(cudaMemcpy(result.data() + sampleStart,
                          dSampleCounts,
                          sizeof(uint32_t) * pixelCount,
                          cudaMemcpyDeviceToHost));

    // Reset the image(all data is transferred)
    Reset(system);
    return std::move(result);
}