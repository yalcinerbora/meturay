#pragma once

#include <vector>
#include <cstdint>
#include <cuda.h>
#include <algorithm>

#include "RayLib/Vector.h"
#include "RayLib/Constants.h"
#include "RayLib/CudaCheck.h"

#include "DeviceMemory.h"
#include "RNGenerator.h"
#include "BinarySearch.cuh"

class CudaSystem;

// One and Two Dimensional Piecewise-Cosntant
// Distributions
// CPU class is a "Group" class meaning that
// it holds multiple distributions on a single allocation
class PWCDistributionGPU1D
{
    private:
        const float*    gPDF    = nullptr;
        const float*    gCDF    = nullptr;
        uint32_t        count   = 0;

        friend class PWCDistributionGroupCPU2D;
        friend class PWCDistributionGroupCPU1D;

    protected:
    public:
        // Constructors & Destructor
                                    PWCDistributionGPU1D() = default;
        __host__ __device__         PWCDistributionGPU1D(const float* dCDFList,
                                                            const float* dPDFList,
                                                            uint32_t count);
                                    PWCDistributionGPU1D(const PWCDistributionGPU1D&) = default;
                                    PWCDistributionGPU1D(PWCDistributionGPU1D&&) = default;
        PWCDistributionGPU1D&       operator=(const PWCDistributionGPU1D&) = default;
        PWCDistributionGPU1D&       operator=(PWCDistributionGPU1D&&) = default;
                                    ~PWCDistributionGPU1D() = default;

        __device__
        float                   Sample(float& pdf, float& index, RNGeneratorGPUI& rng) const;
        __device__
        float                   Pdf(float index) const;
        __host__ __device__
        uint32_t                Count() const { return count; }
};

class PWCDistributionGPU2D
{
    private:
        PWCDistributionGPU1D            gDistributionY;
        const PWCDistributionGPU1D*     gDistributionsX  = nullptr;
        uint32_t                        width            = 0;
        uint32_t                        height           = 0;

        friend class PWCDistributionGroupCPU2D;

    protected:
    public:
        // Constructors & Destructor
                                    PWCDistributionGPU2D() = default;
        __host__ __device__         PWCDistributionGPU2D(const PWCDistributionGPU1D dDistributionsY,
                                                            const PWCDistributionGPU1D* dDistributionsX,
                                                            uint32_t countX,
                                                            uint32_t countY);
                                    PWCDistributionGPU2D(const PWCDistributionGPU2D&) = default;
                                    PWCDistributionGPU2D(PWCDistributionGPU2D&&) = default;
        PWCDistributionGPU2D&    operator=(const PWCDistributionGPU2D&) = default;
        PWCDistributionGPU2D&    operator=(PWCDistributionGPU2D&&) = default;
                                    ~PWCDistributionGPU2D() = default;

        // Interface
        __device__
        Vector2f                Sample(float& pdf, Vector2f& index, RNGeneratorGPUI& rng) const;
        __device__
        float                   Pdf(const Vector2f& index) const;
        __host__ __device__
        uint32_t                Width() const { return width; }
        __host__ __device__
        uint32_t                Height() const { return height; }
        __host__ __device__
        Vector2ui               WidthHeight() const { return Vector2ui(width, height); }
};

class PWCDistributionGroupCPU1D
{
    public:
        using GPUDistList = std::vector<PWCDistributionGPU1D>;

    private:
        DeviceMemory                                memory;
        std::vector<PWCDistributionGPU1D>        gpuDistributions;

        std::vector<size_t>                         counts;
        std::vector<const float*>                   dPDFs;
        std::vector<const float*>                   dCDFs;

        void                                        GeneratePointers();
        void                                        CopyPDFsConstructCDFs(const std::vector<const float*>& functionDataPtrs,
                                                                          const CudaSystem& system,
                                                                          cudaMemcpyKind copyKind);

    protected:
    public:
        // Constructors & Destructor
                                        PWCDistributionGroupCPU1D() = default;
                                        PWCDistributionGroupCPU1D(const std::vector<std::vector<float>>& functions,
                                                                     const CudaSystem& system);
                                        PWCDistributionGroupCPU1D(const std::vector<const float*>& dFunctions,
                                                                     const std::vector<size_t>& counts,
                                                                     const CudaSystem& system);
                                        PWCDistributionGroupCPU1D(const PWCDistributionGroupCPU1D&) = delete;
                                        PWCDistributionGroupCPU1D(PWCDistributionGroupCPU1D&&) = default;
        PWCDistributionGroupCPU1D&   operator=(const PWCDistributionGroupCPU1D&) = delete;
        PWCDistributionGroupCPU1D&   operator=(PWCDistributionGroupCPU1D&&) = default;
                                        ~PWCDistributionGroupCPU1D() = default;

        const PWCDistributionGPU1D&  DistributionGPU(uint32_t index) const;
        const GPUDistList&              DistributionGPU() const;

        size_t                          UsedCPUMemory() const;
        size_t                          UsedGPUMemory() const;
};

class PWCDistributionGroupCPU2D
{
    public:
        struct DistData2D
        {
            const float*                    dYPDF   = nullptr;
            const float*                    dYCDF   = nullptr;
            std::vector<const float*>       dXPDFs;
            std::vector<const float*>       dXCDFs;

            const PWCDistributionGPU1D*  dXDists = nullptr;
            PWCDistributionGPU1D         yDist;
        };

        using GPUDistList = std::vector<PWCDistributionGPU2D>;

    private:
        DeviceMemory                                memory;

        std::vector<Vector2ui>                      dimensions;
        std::vector<DistData2D>                     distDataList;

        //std::vector<const float*>                   dCDFs;
        //std::vector<const PWCDistributionGPU1D*> dXDistributions;
        std::vector<PWCDistributionGPU2D>        gpuDistributions;

        void                                        Allocate(const std::vector<Vector2ui>& dimensions);

    protected:
    public:
        // Constructors & Destructor
                                        PWCDistributionGroupCPU2D() = default;
                                        PWCDistributionGroupCPU2D(const std::vector<const float*>& dFunctions,
                                                                     const std::vector<Vector2ui>& dimensions,
                                                                     const std::vector<bool>& factorSpherical,
                                                                     const CudaSystem& system);
                                        PWCDistributionGroupCPU2D(const std::vector<std::vector<float>>& functions,
                                                                     const std::vector<Vector2ui>& dimensions,
                                                                     const std::vector<bool>& factorSpherical,
                                                                     const CudaSystem& system);
                                        PWCDistributionGroupCPU2D(const PWCDistributionGroupCPU2D&) = delete;
                                        PWCDistributionGroupCPU2D(PWCDistributionGroupCPU2D&&) = default;
        PWCDistributionGroupCPU2D&   operator=(const PWCDistributionGroupCPU2D&) = delete;
        PWCDistributionGroupCPU2D&   operator=(PWCDistributionGroupCPU2D&&) = default;
                                        ~PWCDistributionGroupCPU2D() = default;

        const PWCDistributionGPU2D&  DistributionGPU(uint32_t index) const;
        const GPUDistList&              DistributionGPU() const;

        void                            UpdateDistributions(const std::vector<const float*>& functionDataPtrs,
                                                            const std::vector<bool>& factorSpherical,
                                                            const CudaSystem& system, cudaMemcpyKind kind);

        size_t                          UsedCPUMemory() const;
        size_t                          UsedGPUMemory() const;
};

// Piecewise-Constant Distributions
// GPU-CPU clases
// These classes are the specialized version
// in which each distribution has the same sized data (N by M).
// This implementation has algorithmic improvements
// and it is faster when number of distributions is high (1k+)

class PWCDistStaticGPU2D
{
    private:
        const float*    gXPDFs;
        const float*    gXCDFs;

        const float*    gYCDFs;
        const float*    gYPDFs;

        Vector2ui       dim;
        uint32_t        distCount;

        friend class PWCDistStaticCPU2D;

    public:
        // Interface
        __device__
        Vector2f                Sample(float& pdf, Vector2f& index,
                                       RNGeneratorGPUI& rng,
                                       uint32_t distIndex) const;
        __device__
        float                   Pdf(const Vector2f& index,
                                    uint32_t distIndex) const;
        __host__ __device__
        uint32_t                Width() const { return dim[0]; }
        __host__ __device__
        uint32_t                Height() const { return dim[1]; }
        __host__ __device__
        Vector2ui               WidthHeight() const { return dim; }
};

class PWCDistStaticCPU2D
{
    private:
        PWCDistStaticGPU2D  gpuDist;
        DeviceMemory        memory;

        void                Allocate(uint32_t distCount,
                                     const Vector2ui& dimensions);

    public:
        // Constructors & Destructor
                                    PWCDistStaticCPU2D() = default;
                                    PWCDistStaticCPU2D(const float* dFunctions,
                                                        uint32_t distCount,
                                                        Vector2ui dimensions,
                                                        bool factorSpherical,
                                                        const CudaSystem& system);
                                    PWCDistStaticCPU2D(std::vector<float>& functions,
                                                        uint32_t distCount,
                                                        Vector2ui dimensions,
                                                        bool factorSpherical,
                                                        const CudaSystem& system);
                                    PWCDistStaticCPU2D(const PWCDistStaticCPU2D&) = delete;
                                    PWCDistStaticCPU2D(PWCDistStaticCPU2D&&) = default;
        PWCDistStaticCPU2D&        operator=(const PWCDistStaticCPU2D&) = delete;
        PWCDistStaticCPU2D&        operator=(PWCDistStaticCPU2D&&) = default;
                                    ~PWCDistStaticCPU2D() = default;

        PWCDistStaticGPU2D          DistributionGPU() const;
        // Update Function can be called with device or host pointers
        // with the appropirate memcpy enumeration
        void                        UpdateDistributions(const float* functionData,
                                                        bool factorSpherical,
                                                        const CudaSystem& system,
                                                        cudaMemcpyKind kind);

        size_t                      UsedCPUMemory() const;
        size_t                      UsedGPUMemory() const;
};

__host__ __device__ HYBRID_INLINE
PWCDistributionGPU1D::PWCDistributionGPU1D(const float* dCDFList,
                                                 const float* dPDFList,
                                                 uint32_t count)
    : gPDF(dPDFList)
    , gCDF(dCDFList)
    , count(count)
{}

__device__ inline
float PWCDistributionGPU1D::Sample(float& pdf, float& index, RNGeneratorGPUI& rng) const
{
    float xi = rng.Uniform();

    GPUFunctions::BinarySearchInBetween<float>(index, xi, gCDF, count);
    uint32_t indexInt = static_cast<uint32_t>(index);

    // Extremely rarely index becomes the light count
    // although Uniform should return [0, 1)
    // it still happens due to fp error i guess?
    // if it happens just return the last light on the list
    if(indexInt == count)
    {
        KERNEL_DEBUG_LOG("CUDA Error: Illegal Index on PwC Sample\n");
        index--;
    }

    pdf = gPDF[indexInt];
    return index / float(count);
}

__device__ inline
float PWCDistributionGPU1D::Pdf(float index) const
{
    uint32_t indexInt = static_cast<uint32_t>(index);
    return gPDF[indexInt];
}

__host__ __device__ HYBRID_INLINE
PWCDistributionGPU2D::PWCDistributionGPU2D(const PWCDistributionGPU1D dDistributionY,
                                                 const PWCDistributionGPU1D* dDistributionsX,
                                                 uint32_t countX,
                                                 uint32_t countY)
    : gDistributionY(dDistributionY)
    , gDistributionsX(dDistributionsX)
    , width(countX)
    , height(countY)
{}

__device__ inline
Vector2f PWCDistributionGPU2D::Sample(float& pdf, Vector2f& index, RNGeneratorGPUI& rng) const
{
    // Fist select a row using Y distribution
    float pdfY, indexY;
    float xiY = gDistributionY.Sample(pdfY, indexY, rng);
    uint32_t indexYInt = static_cast<uint32_t>(indexY);

    // Now select column using X distribution of that row
    float pdfX, indexX;
    float xiX = gDistributionsX[indexYInt].Sample(pdfX, indexX, rng);

    // Combined PDF is multiplication since SampleX depends on SampleY
    pdf = pdfX * pdfY;
    index = Vector2f(indexX, indexY);

    return Vector2(xiX, xiY);
}

__device__ inline
float PWCDistributionGPU2D::Pdf(const Vector2f& index) const
{
    int indexY = static_cast<uint32_t>(index[1]);

    float pdfY = gDistributionY.Pdf(index[1]);
    float pdfX = gDistributionsX[indexY].Pdf(index[0]);
    return pdfX * pdfY;
}

__device__ inline
Vector2f PWCDistStaticGPU2D::Sample(float& pdf, Vector2f& index,
                                    RNGeneratorGPUI& rng,
                                    uint32_t distIndex) const
{
    const float* gYCDF = gYCDFs + (dim[1] + 1) * distIndex;
    const float* gYPDF = gYPDFs + dim[1] * distIndex;

    Vector2f xi = Vector2f(rng.Uniform(), rng.Uniform());
    GPUFunctions::BinarySearchInBetween<float>(index[1], xi[1], gYCDFs, dim[1] + 1);
    uint32_t indexYInt = static_cast<uint32_t>(index[1]);
    // Extremely rarely index becomes the light count
    // although Uniform should return [0, 1)
    // it still happens due to fp error i guess?
    // if it happens just return the last light on the list
    if(indexYInt == dim[1] + 1)
    {
        KERNEL_DEBUG_LOG("CUDA Error: Illegal Index on PwC Sample\n");
        indexYInt--;
    }
    // Now sample the X
    const float* gXCDF = gXCDFs + (dim.Multiply() + dim[0]) * distIndex;
    const float* gXCDFRow = gXCDF + (dim[0] + 1) * indexYInt;
    const float* gXPDF = gXPDFs + dim.Multiply() * distIndex;
    const float* gXPDFRow = gXPDF + dim[0] * indexYInt;

    GPUFunctions::BinarySearchInBetween<float>(index[0], xi[0], gXCDFRow, dim[0] + 1);
    uint32_t indexXInt = static_cast<uint32_t>(index[0]);

    pdf = gYPDF[indexYInt] * gXPDFRow[indexXInt];
    return index / Vector2f(static_cast<float>(dim[0]),
                            static_cast<float>(dim[1]));
}

__device__ inline
float PWCDistStaticGPU2D::Pdf(const Vector2f& index,
                              uint32_t distIndex) const
{
    Vector2ui indexInt = Vector2ui(static_cast<uint32_t>(index[0]),
                                   static_cast<uint32_t>(index[1]));


    const float* gYPDF = gYPDFs + dim[1] * distIndex;
    const float* gXPDF = gXPDFs + dim.Multiply() * distIndex;
    const float* gXPDFRow = gXPDF + dim[0] * indexInt[1];

    float pdfX = gXPDFRow[indexInt[0]];
    float pdfY = gYPDF[indexInt[1]];

    float result = pdfX * pdfY;

    if(isnan(result))
    {
        printf("Nan pdf %u => (%f = %f[X] * %f[Y])\n",
               distIndex, result, pdfX, pdfY);
    }

    return result;
}

inline size_t PWCDistributionGroupCPU1D::UsedCPUMemory() const
{
    return (gpuDistributions.size() * sizeof(PWCDistributionGPU1D) +
            counts.size() * sizeof(size_t) +
            dPDFs.size() * sizeof(float*) +
            dCDFs.size() * sizeof(float*));
}

inline size_t PWCDistributionGroupCPU1D::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t PWCDistributionGroupCPU2D::UsedCPUMemory() const
{
    return (dimensions.size() * sizeof(Vector2ui) +
            distDataList.size() * sizeof(DistData2D) +
            //dCDFs.size() * sizeof(float*) +
            //dXDistributions.size() * sizeof(PWCDistributionGPU1D*) +
            gpuDistributions.size() * sizeof(PWCDistributionGPU2D));
}

inline size_t PWCDistributionGroupCPU2D::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t PWCDistStaticCPU2D::UsedCPUMemory() const
{
    return sizeof(PWCDistStaticGPU2D);
}

inline size_t PWCDistStaticCPU2D::UsedGPUMemory() const
{
    return memory.Size();
}