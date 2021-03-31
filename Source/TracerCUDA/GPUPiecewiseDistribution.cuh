#pragma once

#include <vector>
#include <cstdint>
#include <cuda.h>
#include <algorithm>
#include "RayLib/Vector.h"

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
        float                   Sample(float& pdf, RandomGPU& rng) const;
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
        Vector2f                Sample(float& pdf, RandomGPU& rng) const;
        __host__ __device__
        uint32_t                Width() const { return width; }
        __host__ __device__
        uint32_t                Height() const { return height; }
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
        std::vector<const GPUDistPiecewiseConst1D*> dXDistributions;

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
                                                                const CudaSystem& system);
                                        CPUDistGroupPiecewiseConst2D(const CPUDistGroupPiecewiseConst2D&) = delete;
                                        CPUDistGroupPiecewiseConst2D(CPUDistGroupPiecewiseConst2D&&) = default;
        CPUDistGroupPiecewiseConst2D&   operator=(const CPUDistGroupPiecewiseConst2D&) = delete;
        CPUDistGroupPiecewiseConst2D&   operator=(CPUDistGroupPiecewiseConst2D&&) = default;
                                        ~CPUDistGroupPiecewiseConst2D() = default;

        const GPUDistPiecewiseConst2D&  DistributionGPU(uint32_t index) const;
        const GPUDistList&              DistributionGPU() const;
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
inline float GPUDistPiecewiseConst1D::Sample(float& pdf, RandomGPU& rng) const
{
    float xi = GPUDistribution::Uniform<float>(rng);
    float index;
    GPUFunctions::BinarySearchInBetween<float>(index, xi, gCDF, count);
    pdf = gPDF[static_cast<uint32_t>(index)];
    return index;
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
inline Vector2f GPUDistPiecewiseConst2D::Sample(float& pdf, RandomGPU& rng) const
{
    // Fist select a row using Y distribution
    float pdfY;
    float indexY = gDistributionsY.Sample(pdfY, rng);

    // Now select column using X distribution of that row
    float pdfX;
    float indexX = gDistributionsX[static_cast<uint32_t>(indexY)].Sample(pdfX, rng);

    // Combined PDF is multiplication since SampleX depends on SampleY
    pdf = pdfX * pdfY;
    return Vector2(indexX, indexY);
}