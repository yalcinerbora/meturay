#pragma once

#include <vector>
#include <cstdint>
#include <cuda.h>
#include <algorithm>
#include "RayLib/Vector.h"

#include "DeviceMemory.h"
#include "Random.cuh"
#include "BinarySearch.cuh"

class GPUDistPiecewise1D
{
    private:
        const float*    gPDF    = nullptr;
        const float*    gCDF    = nullptr;
        uint32_t        count   = 0;

    protected:
    public:
        // Construtors & Destructor
                                //GPUDistPiecewise1D() = default;
        __host__ __device__     GPUDistPiecewise1D(const float* dCDFList,
                                                   const float* dPDFList,
                                                   uint32_t count);
                                GPUDistPiecewise1D(const GPUDistPiecewise1D&) = default;
                                GPUDistPiecewise1D(GPUDistPiecewise1D&&) = default;
        GPUDistPiecewise1D&     operator=(const GPUDistPiecewise1D&) = default;
        GPUDistPiecewise1D&     operator=(GPUDistPiecewise1D&&) = default;
                                ~GPUDistPiecewise1D() = default;

        __host__ __device__
        float                   Sample(float& pdf, RandomGPU& rng) const;
        __host__ __device__
        uint32_t                Count() const { return count; }
};

class GPUDistPiecewise2D
{
    private:
        const GPUDistPiecewise1D   gDistributionsY;
        const GPUDistPiecewise1D*  gDistributionsX  = nullptr;
        uint32_t                   width            = 0;
        uint32_t                   height           = 0;

    protected:
    public:
        // Construtors & Destructor
                                //GPUDistPiecewise2D() = default;
        __host__ __device__     GPUDistPiecewise2D(const GPUDistPiecewise1D dDistributionsY,
                                                   const GPUDistPiecewise1D* dDistributionsX,
                                                   uint32_t countX,
                                                   uint32_t countY);
                                GPUDistPiecewise2D(const GPUDistPiecewise2D&) = default;
                                GPUDistPiecewise2D(GPUDistPiecewise2D&&) = default;
        GPUDistPiecewise2D&     operator=(const GPUDistPiecewise2D&) = default;
        GPUDistPiecewise2D&     operator=(GPUDistPiecewise2D&&) = default;
                                ~GPUDistPiecewise2D() = default;

        // Interface                                
        __host__ __device__ 
        Vector2f                Sample(float& pdf, RandomGPU& rng) const;
        __host__ __device__ 
        uint32_t                Width() const { return width; }
        uint32_t                Height() const { return height; }
};

class CPUDistGroupPiecewise1D
{
    private:
        DeviceMemory                            memory;
        std::vector<GPUDistPiecewise1D>         gpuDistributions;

        std::vector<size_t>                     counts;
        std::vector<const float*>               dPDFs;
        std::vector<const float*>               dCDFs;
        std::vector<const GPUDistPiecewise1D*>  dXDistributions;

    protected:
    public:
        // Constructors & Destructor
                                        CPUDistGroupPiecewise1D() = default;
                                        CPUDistGroupPiecewise1D(const std::vector<std::vector<float>>& pdfValues,
                                                                const CudaSystem& system);
                                        CPUDistGroupPiecewise1D(const CPUDistGroupPiecewise1D&) = delete;
                                        CPUDistGroupPiecewise1D(CPUDistGroupPiecewise1D&&) = default;
        CPUDistGroupPiecewise1D&        operator=(const CPUDistGroupPiecewise1D&) = delete;
        CPUDistGroupPiecewise1D&        operator=(CPUDistGroupPiecewise1D&&) = default;
                                        ~CPUDistGroupPiecewise1D() = default;

        const GPUDistPiecewise1D&       DistributionGPU(uint32_t index) const;
};

class CPUDistGroupPiecewise2D
{
    private:
        DeviceMemory                    memory;
        std::vector<GPUDistPiecewise2D> gpuDistributions;

        std::vector<Vector2ui>          dimensions;
        std::vector<const float*>       dPDFs;
        std::vector<const float*>       dCDFs;

    protected:
    public:
        // Constructors & Destructor
                                        CPUDistGroupPiecewise2D() = default;
                                        CPUDistGroupPiecewise2D(const std::vector<std::vector<float>>& pdfValues,
                                                                const std::vector<Vector2ui>& dimensions,
                                                                const CudaSystem& system);
                                        CPUDistGroupPiecewise2D(const CPUDistGroupPiecewise2D&) = delete;
                                        CPUDistGroupPiecewise2D(CPUDistGroupPiecewise2D&&) = default;
        CPUDistGroupPiecewise2D&        operator=(const CPUDistGroupPiecewise2D&) = delete;
        CPUDistGroupPiecewise2D&        operator=(CPUDistGroupPiecewise2D&&) = default;
                                        ~CPUDistGroupPiecewise2D() = default;
       
        const GPUDistPiecewise2D&       DistributionGPU(uint32_t index) const;
};

__host__
inline GPUDistPiecewise1D::GPUDistPiecewise1D(const float* dCDFList,
                                              const float* dPDFList,
                                              uint32_t count)
    : gCDF(dCDFList)
    , gPDF(dPDFList)
    , count(count)
{}

__host__
inline float GPUDistPiecewise1D::Sample(float& pdf, RandomGPU& rng) const
{
    float xi = GPUDistribution::Uniform<float>(rng);
    float index;
    GPUFunctions::BinarySearchInBetween<float>(index, xi, gCDF, count);
    pdf = gPDF[static_cast<uint32_t>(index)];
    return index;
}

__host__
inline GPUDistPiecewise2D::GPUDistPiecewise2D(const GPUDistPiecewise1D dDistributionsY,
                                              const GPUDistPiecewise1D* dDistributionsX,
                                              uint32_t countX,
                                              uint32_t countY)
    : gDistributionsY(dDistributionsY)
    , gDistributionsX(dDistributionsX)
    , width(countX)
    , height(countY)
{}

__device__
inline Vector2f GPUDistPiecewise2D::Sample(float& pdf, RandomGPU& rng) const
{
    // Fist select a row using Y distribution
    float pdfY;
    float indexY = gDistributionsY.Sample(pdf, rng);

    // Now select column using X distribution of that row
    float pdfX;
    float indexX = gDistributionsX[static_cast<uint32_t>(indexX)].Sample(pdfY, rng);

    // Combined PDF is multiplication since SampleX depends on SampleY
    pdf = pdfX * pdfY;
    return Vector2(indexX, indexY);
}