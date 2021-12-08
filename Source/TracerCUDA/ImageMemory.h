#pragma once
/**

Basic output image memory
Used by tracers

*/

#include <vector>

#include "RayLib/Types.h"
#include "RayLib/Vector.h"

#include "DeviceMemory.h"
#include "ImageStructs.h"

class CudaSystem;

class ImageMemory
{
    private:
        DeviceMemory         memory;
        PixelFormat          format;
        int                  pixelSize;

        Vector2i             segmentSize;
        Vector2i             segmentOffset;
        Vector2i             resolution;

        void*                dPixels;
        uint32_t*            dSampleCounts;

    protected:
    public:
        // Constructors & Destructor
                            ImageMemory();
                            ImageMemory(const Vector2i& offset,
                                        const Vector2i& size,
                                        const Vector2i& resolution,
                                        PixelFormat f);
                            ImageMemory(const ImageMemory&) = delete;
                            ImageMemory(ImageMemory&&) = default;
        ImageMemory&        operator=(const ImageMemory&) = delete;
        ImageMemory&        operator=(ImageMemory&&) = default;
                            ~ImageMemory() = default;

        void                SetPixelFormat(PixelFormat, const CudaSystem& s);
        void                Reportion(Vector2i start,
                                      Vector2i end,
                                      const CudaSystem& system);
        void                Resize(Vector2i resolution);
        void                Reset(const CudaSystem& system);

        // Getters
        Vector2i            SegmentSize() const;
        Vector2i            SegmentOffset() const;
        Vector2i            Resolution() const;
        PixelFormat         Format() const;
        int                 PixelSize() const;

        // Image Global Memory
        template <class T>
        ImageGMem<T>        GMem();
        template <class T>
        ImageGMemConst<T>   GMem() const;

        // Memory Usage
        size_t              UsedGPUMemory() const;

        // Direct CPU
        std::vector<Byte>   GetImageToCPU(const CudaSystem&);
};

inline Vector2i ImageMemory::SegmentSize() const
{
    return segmentSize;
}

inline Vector2i ImageMemory::SegmentOffset() const
{
    return segmentOffset;
}

inline Vector2i ImageMemory::Resolution() const
{
    return resolution;
}

inline PixelFormat ImageMemory::Format() const
{
    return format;
}

inline int ImageMemory::PixelSize() const
{
    return pixelSize;
}

template<class T>
inline ImageGMem<T> ImageMemory::GMem()
{
    return ImageGMem<T>{static_cast<T*>(dPixels), dSampleCounts};
}

template<class T>
inline ImageGMemConst<T> ImageMemory::GMem() const
{
    return ImageGMemConst<T>{static_cast<T*>(dPixels), dSampleCounts};
}

inline size_t ImageMemory::UsedGPUMemory() const
{
    return memory.Size();
}