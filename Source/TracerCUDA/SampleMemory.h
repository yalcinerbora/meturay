#pragma once

#include "RayLib/Types.h"
#include "RayLib/Vector.h"

#include "ImageIO/ImageIOI.h"

#include "DeviceMemory.h"
#include "ImageStructs.h"

class SampleMemory
{
    private:
        DeviceMemory        memory;
        PixelFormat         format;

        uint32_t            sampleCount;

        void*               dValues;
        Vector2f*           dImgCoords;

    protected:
    public:
        // Constructors & Destructor
                                SampleMemory();
                                SampleMemory(uint32_t sampleCount,
                                            PixelFormat f);
                                SampleMemory(const SampleMemory&) = delete;
                                SampleMemory(SampleMemory&&) = default;
        SampleMemory&           operator=(const SampleMemory&) = delete;
        SampleMemory&           operator=(SampleMemory&&) = default;
                                ~SampleMemory() = default;

        void                    Resize(PixelFormat f, uint32_t newCount);
        void                    Reset();

        // Sample Global Memory Access
        template <class T>
        CamSampleGMem<T>        GMem();
        template <class T>
        CamSampleGMemConst<T>   GMem() const;

        size_t                  UsedGPUMemory() const;
};

inline SampleMemory::SampleMemory()
    : format(PixelFormat::END)
    , sampleCount(0)
    , dValues(nullptr)
    , dImgCoords(nullptr)
{}

inline SampleMemory::SampleMemory(uint32_t sampleCount,
                                  PixelFormat f)
    : format(PixelFormat::END)
    , sampleCount(sampleCount)
    , dValues(nullptr)
    , dImgCoords(nullptr)
{
    Resize(f, sampleCount);
    Reset();
}

inline void SampleMemory::Resize(PixelFormat f, uint32_t newCount)
{
    format = f;
    size_t pixelSize = ImageIOI::FormatToPixelSize(f);
    size_t sizeOfValues = pixelSize * newCount;
    sizeOfValues = Memory::AlignSize(sizeOfValues);
    size_t sizeOfImgCoords = newCount * sizeof(Vector2f);

    if(newCount != 0)
    {
        GPUMemFuncs::EnlargeBuffer(memory, sizeOfValues + sizeOfImgCoords);

        size_t offset = 0;
        std::uint8_t* dMem = static_cast<uint8_t*>(memory);
        dValues = dMem + offset;
        offset += sizeOfValues;
        dImgCoords = reinterpret_cast<Vector2f*>(dMem + offset);
        offset += sizeOfImgCoords;
        assert(offset == (sizeOfValues + sizeOfImgCoords));
    }
    sampleCount = newCount;
}

inline void SampleMemory::Reset()
{
    if(sampleCount != 0)
    {
        size_t totalBytes = memory.Size();
        CUDA_CHECK(cudaMemset(memory, 0x00, totalBytes));
    }
}

template <class T>
inline CamSampleGMem<T> SampleMemory::GMem()
{
    return CamSampleGMem<T>
    {
        reinterpret_cast<T*>(dValues),
        dImgCoords
    };
}
template <class T>
inline CamSampleGMemConst<T> SampleMemory::GMem() const
{
    return CamSampleGMemConst<T>
    {
        reinterpret_cast<const T*>(dValues),
        dImgCoords
    };
}

inline size_t SampleMemory::UsedGPUMemory() const
{
    return memory.Size();
}