#pragma once
/**

Compressed Bitmap Implementation

*/

#include "RayLib/Types.h"
#include "RayLib/Vector.h"
#include "RayLib/MemoryAlignment.h"

#include "DeviceMemory.h"
#include <vector>

class GPUBitmap
{
    private:
        const Byte*         gBools;
        Vector2ui           dimensions;

    protected:

    public:
        // Constructors & Destructor
                            GPUBitmap();
                            GPUBitmap(const Byte* dVals,
                                      const Vector2ui& dim);
                            ~GPUBitmap() = default;


        __device__ bool     operator()(const Vector2f& uv) const;
};

class CPUBitmapGroup
{
    private:
        DeviceMemory            memory;
        const GPUBitmap*        dGPUBitmaps;
        std::vector<Vector2ui>  dimensions;

    protected:
    public:
        // Constructors & Destructor
                                CPUBitmapGroup() = default;
                                CPUBitmapGroup(const std::vector<std::vector<Byte>>& bits,
                                               const std::vector<Vector2ui>& dimensions);
                                CPUBitmapGroup(const CPUBitmapGroup&) = default;
                                CPUBitmapGroup(CPUBitmapGroup&&) = default;
        CPUBitmapGroup&         operator=(const CPUBitmapGroup&) = default;
        CPUBitmapGroup&         operator=(CPUBitmapGroup&&) = default;
                                ~CPUBitmapGroup() = default;

        const GPUBitmap*        AllBitmaps() const;
        const GPUBitmap*        Bitmap(uint32_t index) const;

};

inline GPUBitmap::GPUBitmap()
    : gBools(nullptr)
    , dimensions(Zero2ui)
{}

inline GPUBitmap::GPUBitmap(const Byte* dVals,
                            const Vector2ui& dim)
    : gBools(dVals)
    , dimensions(dim)
{}

__device__
inline bool GPUBitmap::operator()(const Vector2f& uv) const
{
    // Wrap Texture
    Vector2f uvNorm = Vector2f(uv[0] - floorf(uv[0]),
                               uv[1] - floorf(uv[1]));

    Vector2ui index = Vector2ui(uvNorm * Vector2f(dimensions));
    uint32_t linearSize = index[1] * dimensions[0] + index[0];
    uint32_t byteIndex = linearSize / BYTE_BITS;
    uint32_t innerIndex = linearSize % BYTE_BITS;
    bool opaque = static_cast<bool>((gBools[byteIndex] >> innerIndex) & 0x01);
    return opaque;
}

inline CPUBitmapGroup::CPUBitmapGroup(const std::vector<std::vector<Byte>>& bits,
                                      const std::vector<Vector2ui>& dimensions)
    : dimensions(dimensions)
    , dGPUBitmaps(nullptr)
{
    if(dimensions.size() == 0) return;

    size_t totalSize = 0;
    std::vector<size_t> allocationOffsets;
    allocationOffsets.reserve(bits.size() + 1);
    for(const auto& bit : bits)
    {
        size_t allocSize = bit.size() * sizeof(Byte);
        allocSize = Memory::AlignSize(allocSize);
        allocationOffsets.push_back(totalSize);
        totalSize += allocSize;
    }
    allocationOffsets.push_back(totalSize);
    // Calculate GPUBitmap Size
    size_t bitmapClassSize = bits.size() * sizeof(GPUBitmap);
    bitmapClassSize = Memory::AlignSize(bitmapClassSize);
    totalSize += bitmapClassSize;

    // Allocation
    memory = DeviceMemory(totalSize);

    std::vector<GPUBitmap> hGPUBitmaps;
    hGPUBitmaps.reserve(bits.size());
    for(uint32_t i = 0; i < allocationOffsets.size() - 1; i++)
    {
        size_t offset = allocationOffsets[i];
        Byte* dPtr = static_cast<Byte*>(memory) + offset;

        CUDA_CHECK(cudaMemcpy(dPtr, bits[i].data(),
                              bits[i].size() * sizeof(Byte),
                              cudaMemcpyHostToDevice));

        hGPUBitmaps.emplace_back(dPtr, dimensions[i]);
    }
    dGPUBitmaps = reinterpret_cast<GPUBitmap*>(static_cast<Byte*>(memory) + allocationOffsets.back());
    // Copy GPU Bitmaps to GPU
    CUDA_CHECK(cudaMemcpy(const_cast<GPUBitmap*>(dGPUBitmaps), hGPUBitmaps.data(),
                          hGPUBitmaps.size() * sizeof(GPUBitmap),
                          cudaMemcpyHostToDevice));
}

inline const GPUBitmap* CPUBitmapGroup::AllBitmaps() const
{
    return dGPUBitmaps;
}

inline const GPUBitmap* CPUBitmapGroup::Bitmap(uint32_t index) const
{
    return dGPUBitmaps + index;
}