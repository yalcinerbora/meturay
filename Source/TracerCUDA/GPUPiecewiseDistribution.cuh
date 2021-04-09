#pragma once

#include <vector>
#include <cstdint>
#include <cuda.h>
#include <algorithm>

#include "RayLib/Vector.h"
#include "RayLib/Constants.h"

#include "DeviceMemory.h"
#include "Random.cuh"
#include "BinarySearch.cuh"

class GPUDistPiecewiseConst1D
{
    private:
        const float*    gPDF    = nullptr;
        const float*    gCDF    = nullptr;
        uint32_t        count   = 0;

    protected:
    public:
        // Construtors & Destructor
                                    GPUDistPiecewiseConst1D() = default;
        __host__ __device__         GPUDistPiecewiseConst1D(const float* dCDFList,
                                                            const float* dPDFList,
                                                            uint32_t count);
                                    GPUDistPiecewiseConst1D(const GPUDistPiecewiseConst1D&) = default;
                                    GPUDistPiecewiseConst1D(GPUDistPiecewiseConst1D&&) = default;
        GPUDistPiecewiseConst1D&    operator=(const GPUDistPiecewiseConst1D&) = default;
        GPUDistPiecewiseConst1D&    operator=(GPUDistPiecewiseConst1D&&) = default;
                                    ~GPUDistPiecewiseConst1D() = default;

        __device__
        float                   Sample(float& pdf, float& index, RandomGPU& rng) const;
        __host__ __device__
        uint32_t                Count() const { return count; }
};

class GPUDistPiecewiseConst2D
{
    private:
        const GPUDistPiecewiseConst1D   gDistributionsY;
        const GPUDistPiecewiseConst1D*  gDistributionsX  = nullptr;
        uint32_t                        width            = 0;
        uint32_t                        height           = 0;

    protected:
    public:
        // Construtors & Destructor
                                    GPUDistPiecewiseConst2D() = default;
        __host__ __device__         GPUDistPiecewiseConst2D(const GPUDistPiecewiseConst1D dDistributionsY,
                                                            const GPUDistPiecewiseConst1D* dDistributionsX,
                                                            uint32_t countX,
                                                            uint32_t countY);
                                    GPUDistPiecewiseConst2D(const GPUDistPiecewiseConst2D&) = default;
                                    GPUDistPiecewiseConst2D(GPUDistPiecewiseConst2D&&) = default;
        GPUDistPiecewiseConst2D&    operator=(const GPUDistPiecewiseConst2D&) = default;
        GPUDistPiecewiseConst2D&    operator=(GPUDistPiecewiseConst2D&&) = default;
                                    ~GPUDistPiecewiseConst2D() = default;

        // Interface
        __device__
        Vector2f                Sample(float& pdf, Vector2f& index, RandomGPU& rng) const;
        __host__ __device__
        uint32_t                Width() const { return width; }
        __host__ __device__
        uint32_t                Height() const { return height; }
        __host__ __device__
        Vector2ui               WidthHeight() const { return Vector2ui(width, height); }
};

class CPUDistGroupPiecewiseConst1D
{
    public:
        using GPUDistList = std::vector<GPUDistPiecewiseConst1D>;

    private:
        DeviceMemory                                memory;
        std::vector<GPUDistPiecewiseConst1D>        gpuDistributions;

        std::vector<size_t>                         counts;
        std::vector<const float*>                   dPDFs;
        std::vector<const float*>                   dCDFs;

    protected:
    public:
        // Constructors & Destructor
                                        CPUDistGroupPiecewiseConst1D() = default;
                                        CPUDistGroupPiecewiseConst1D(const std::vector<std::vector<float>>& functions,
                                                                     const CudaSystem& system);
                                        CPUDistGroupPiecewiseConst1D(const CPUDistGroupPiecewiseConst1D&) = delete;
                                        CPUDistGroupPiecewiseConst1D(CPUDistGroupPiecewiseConst1D&&) = default;
        CPUDistGroupPiecewiseConst1D&   operator=(const CPUDistGroupPiecewiseConst1D&) = delete;
        CPUDistGroupPiecewiseConst1D&   operator=(CPUDistGroupPiecewiseConst1D&&) = default;
                                        ~CPUDistGroupPiecewiseConst1D() = default;

        const GPUDistPiecewiseConst1D&  DistributionGPU(uint32_t index) const;
        const GPUDistList&              DistributionGPU() const;

        size_t                          UsedCPUMemory() const;
        size_t                          UsedGPUMemory() const;
};

class CPUDistGroupPiecewiseConst2D
{
    public:
        struct DistData2D
        {
            const float*                    dYPDF   = nullptr;
            const float*                    dYCDF   = nullptr;
            std::vector<const float*>       dXPDFs;
            std::vector<const float*>       dXCDFs;

            const GPUDistPiecewiseConst1D*  dXDists = nullptr;
            GPUDistPiecewiseConst1D         yDist;
        };

        using GPUDistList = std::vector<GPUDistPiecewiseConst2D>;

    private:
        DeviceMemory                                memory;

        std::vector<Vector2ui>                      dimensions;
        std::vector<DistData2D>                     distDataList;

        std::vector<const float*>                   dCDFs;
        std::vector<const GPUDistPiecewiseConst1D*> dXDistributions;
        std::vector<GPUDistPiecewiseConst2D>        gpuDistributions;

    protected:
    public:
        // Constructors & Destructor
                                        CPUDistGroupPiecewiseConst2D() = default;
                                        CPUDistGroupPiecewiseConst2D(const std::vector<std::vector<float>>& functions,
                                                                     const std::vector<Vector2ui>& dimensions,
                                                                     const std::vector<bool>& factorSpherical,
                                                                     const CudaSystem& system);
                                        CPUDistGroupPiecewiseConst2D(const CPUDistGroupPiecewiseConst2D&) = delete;
                                        CPUDistGroupPiecewiseConst2D(CPUDistGroupPiecewiseConst2D&&) = default;
        CPUDistGroupPiecewiseConst2D&   operator=(const CPUDistGroupPiecewiseConst2D&) = delete;
        CPUDistGroupPiecewiseConst2D&   operator=(CPUDistGroupPiecewiseConst2D&&) = default;
                                        ~CPUDistGroupPiecewiseConst2D() = default;

        const GPUDistPiecewiseConst2D&  DistributionGPU(uint32_t index) const;
        const GPUDistList&              DistributionGPU() const;

        size_t                          UsedCPUMemory() const;
        size_t                          UsedGPUMemory() const;
};

__host__ __device__
inline GPUDistPiecewiseConst1D::GPUDistPiecewiseConst1D(const float* dCDFList,
                                                        const float* dPDFList,
                                                        uint32_t count)
    : gCDF(dCDFList)
    , gPDF(dPDFList)
    , count(count)
{}

__device__
inline float GPUDistPiecewiseConst1D::Sample(float& pdf, float& index, RandomGPU& rng) const
{
    float xi = GPUDistribution::Uniform<float>(rng);
    // Extremely rarely index becomes the light count
    // although GPUDistribution::Uniform should return [0, 1)
    // it still happens due to fp error i guess?
    // if it happens just return the last light on the list
    //if(xi == 1.0f) xi -= MathConstants::VeryLargeEpsilon;

    GPUFunctions::BinarySearchInBetween<float>(index, xi, gCDF, count);
    uint32_t indexInt = static_cast<uint32_t>(index);

    if(indexInt == count)
    {
        printf("CUDA Error: Illegal Index on PwC Sample\n");
        index--;
    }

    pdf = gPDF[indexInt];
    return index / float(count);
}

__host__ __device__
inline GPUDistPiecewiseConst2D::GPUDistPiecewiseConst2D(const GPUDistPiecewiseConst1D dDistributionsY,
                                                        const GPUDistPiecewiseConst1D* dDistributionsX,
                                                        uint32_t countX,
                                                        uint32_t countY)
    : gDistributionsY(dDistributionsY)
    , gDistributionsX(dDistributionsX)
    , width(countX)
    , height(countY)
{}

__device__
inline Vector2f GPUDistPiecewiseConst2D::Sample(float& pdf, Vector2f& index, RandomGPU& rng) const
{
    // Fist select a row using Y distribution
    float pdfY, indexY;
    float xiY = gDistributionsY.Sample(pdfY, indexY, rng);
    uint32_t indexYInt = static_cast<uint32_t>(indexY);

    // Now select column using X distribution of that row
    float pdfX, indexX;
    float xiX = gDistributionsX[indexYInt].Sample(pdfX, indexX, rng);

    // Combined PDF is multiplication since SampleX depends on SampleY
    pdf = pdfX * pdfY;
    index = Vector2f(indexX, indexY);

    //printf("Index (%f, %f) UV (%f, %f)  ", index[0], index[1], xiX, xiY);

    return Vector2(xiX, xiY);
}

inline size_t CPUDistGroupPiecewiseConst1D::UsedCPUMemory() const
{
    return (gpuDistributions.size() * sizeof(GPUDistPiecewiseConst1D) +
            counts.size() * sizeof(size_t) +
            dPDFs.size() * sizeof(float*) +
            dCDFs.size() * sizeof(float*));
}

inline size_t CPUDistGroupPiecewiseConst1D::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPUDistGroupPiecewiseConst2D::UsedCPUMemory() const
{
    return (dimensions.size() * sizeof(Vector2ui) +
            distDataList.size() * sizeof(DistData2D) +
            dCDFs.size() * sizeof(float*) +
            dXDistributions.size() * sizeof(GPUDistPiecewiseConst1D*) +
            gpuDistributions.size() * sizeof(GPUDistPiecewiseConst2D));
}

inline size_t CPUDistGroupPiecewiseConst2D::UsedGPUMemory() const
{
    return memory.Size();
}