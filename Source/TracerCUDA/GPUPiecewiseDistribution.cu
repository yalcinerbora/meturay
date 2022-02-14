#include "GPUPiecewiseDistribution.cuh"
#include "ParallelScan.cuh"
#include "ParallelReduction.cuh"
#include "ParallelTransform.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

#include "RayLib/Types.h"
#include "RayLib/MemoryAlignment.h"

#include <numeric>
#include <cub/cub.cuh>

#include "TracerDebug.h"

template <uint32_t DATA_PER_THREAD>
__device__ __forceinline__
void MultVector(float(&vector)[DATA_PER_THREAD], float v)
{
    #pragma unroll
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
        vector[i] *= v;
}

template <uint32_t DATA_PER_THREAD>
__device__ __forceinline__
void ZeroVector(float(&vector)[DATA_PER_THREAD])
{
    #pragma unroll
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
        vector[i] = 0.0f;
}

__global__
void KCUpdateDistributionsX(float* gXPDFs,
                            float* gXCDFs,
                            float* gYPDFs,
                            Vector2ui dim,
                            uint32_t distCount,
                            float factorSpherical)
{
    static constexpr uint32_t DATA_PER_THREAD = 4;
    assert(StaticThreadPerBlock1D * DATA_PER_THREAD >= dim[0]);
    // Cub Stuff
    // Cub Block Load Specialize
    using BlockLoad = cub::BlockLoad<float, StaticThreadPerBlock1D, DATA_PER_THREAD>;
    using BlockReduce = cub::BlockReduce<float, StaticThreadPerBlock1D>;
    using BlockScan = cub::BlockScan<float, StaticThreadPerBlock1D>;
    using BlockStore = cub::BlockStore<float, StaticThreadPerBlock1D, DATA_PER_THREAD>;
    // Allocate shared memory for BlockLoad
    __shared__ union
    {
        typename BlockLoad::TempStorage      loadTempStorage;
        typename BlockReduce::TempStorage    reduceTempStorage;
        typename BlockScan::TempStorage      scanTempStorage;
        typename BlockStore::TempStorage     storeTempStorage;
    } sMem;

    // Each block is responsible for one dist
    uint32_t distId = blockIdx.x;
    if(distId >= distCount) return;

    // First find the local data pointers
    float* gXPDFLocal = gXPDFs + dim.Multiply() * distId;
    float* gXCDFLocal = gXCDFs + (dim.Multiply() + dim[0]) * distId;
    float* gYPDFLocal = gYPDFs + dim[1] * distId;

    // Local registers
    float pdfData[DATA_PER_THREAD];
    float cdfData[DATA_PER_THREAD];
    // For each row
    for(uint32_t i = 0; i < dim[1]; i++)
    {
        float* gXPDFRow = gXPDFLocal + dim[0] * i;
        float* gXCDFRow = gXCDFLocal + (dim[0] + 1) * i;

        // Load a segment of consecutive items that are blocked across threads
        ZeroVector(pdfData);
        ZeroVector(cdfData);
        BlockLoad(sMem.loadTempStorage).Load(gXPDFRow, pdfData,
                                             static_cast<int>(dim[0]));
        __syncthreads();

        if(factorSpherical)
        {
            float v = (static_cast<float>(i) + 0.5f) / static_cast<float>(dim[1]);
            // V is [0,1] convert it to [0,pi]
            // so that middle sections will have higher priority
            float phi = v * MathConstants::Pi;
            float sinPhi = std::sin(phi);
            MultVector(pdfData, sinPhi);
        }

        // Reduce the row
        float pdfSum = BlockReduce(sMem.reduceTempStorage).Sum(pdfData);
        __syncthreads();

        // First thread will do the write of the Y Function value of this row
        if(threadIdx.x == 0)
            gYPDFLocal[i] = pdfSum;

        // Now normalize the pdf with the dimension
        MultVector(pdfData, (1.0f / static_cast<float>(dim[1])));

        // Now do scan
        float totalSum;
        BlockScan(sMem.scanTempStorage).ExclusiveSum(pdfData, cdfData, totalSum);
        __syncthreads();

        // Do the normalization for PDF and CDF
        MultVector(pdfData, (1.0f / totalSum));
        MultVector(pdfData, static_cast<float>(dim[0]));

        MultVector(cdfData, (1.0f / totalSum));

        // Finally Store both cdf and pdf
        BlockStore(sMem.storeTempStorage).Store(gXPDFRow, pdfData, dim[0]);
        __syncthreads();
        BlockStore(sMem.storeTempStorage).Store(gXCDFRow, cdfData, dim[0]);
        __syncthreads();
        // Thread 0 will write the total sum to the end (normalized which is 1)
        if(threadIdx.x == 0)
            gXCDFRow[dim[0]] = 1.0f;
    }
}

__global__
void KCUpdateDistributionsY(float* gYPDFs,
                            float* gYCDFs,
                            Vector2ui dim,
                            uint32_t distCount)
{
    static constexpr uint32_t DATA_PER_THREAD = 4;
    assert(StaticThreadPerBlock1D * DATA_PER_THREAD >= dim[1]);

    // Cub Stuff
    // Cub Block Load Specialize
    using BlockLoad = cub::BlockLoad<float, StaticThreadPerBlock1D,
                                     DATA_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockScan = cub::BlockScan<float, StaticThreadPerBlock1D>;
    using BlockStore = cub::BlockStore<float, StaticThreadPerBlock1D,
                                       DATA_PER_THREAD, cub::BLOCK_STORE_VECTORIZE>;
    // Allocate shared memory for BlockLoad
    __shared__ union
    {
        typename BlockLoad::TempStorage      loadTempStorage;
        typename BlockScan::TempStorage      scanTempStorage;
        typename BlockStore::TempStorage     storeTempStorage;
    } sMem;

    // Each block is responsible for one dist
    uint32_t distId = blockIdx.x;
    if(distId >= distCount) return;

    // First find the local data pointers
    float* gYPDFLocal = gYPDFs + dim[1] * distId;
    float* gYCDFLocal = gYCDFs + (dim[1] + 1) * distId;

    // Registers
    float pdfData[DATA_PER_THREAD];
    float cdfData[DATA_PER_THREAD];
    ZeroVector(pdfData);
    ZeroVector(cdfData);

    // Load a segment of consecutive items that are blocked across threads
    BlockLoad(sMem.loadTempStorage).Load(gYPDFLocal, pdfData, dim[1]);
    __syncthreads();

    // Normalize by dimension
    MultVector(pdfData, (1.0f / static_cast<float>(dim[1])));

    // Now do scan
    float totalSum;
    BlockScan(sMem.scanTempStorage).ExclusiveSum(pdfData, cdfData, totalSum);
    __syncthreads();

    MultVector(pdfData, (1.0f / totalSum));
    MultVector(pdfData, static_cast<float>(dim[1]));

    MultVector(cdfData, (1.0f / totalSum));

    BlockStore(sMem.storeTempStorage).Store(gYPDFLocal, pdfData, dim[1]);
    __syncthreads();
    BlockStore(sMem.storeTempStorage).Store(gYCDFLocal, cdfData, dim[1]);

    // Thread 0 will write the total sum to the end (normalized which is 1)
    if(threadIdx.x == 0)
        gYCDFLocal[dim[1]] = 1.0f;
}

template <class T>
class DeviceHostMulDivideComboFunctor
{
    private:
        const T&    gDivValue;
        const T     mulValue;

    protected:
    public:
                    DeviceHostMulDivideComboFunctor(const T& dDivValue, T hMulValue);
                    ~DeviceHostMulDivideComboFunctor() = default;

    __device__
    T               operator()(const T& in) const;
};

template <class T>
DeviceHostMulDivideComboFunctor<T>::DeviceHostMulDivideComboFunctor(const T& dDivValue, T hMulValue)
    : gDivValue(dDivValue)
    , mulValue(hMulValue)
{}

template <class T>
__device__  inline
T DeviceHostMulDivideComboFunctor<T>::operator()(const T& in) const
{
    return in * mulValue / gDivValue;
}

void PWCDistributionGroupCPU1D::GeneratePointers()
{
    std::vector<Vector2ui> alignedSizes(counts.size());
    std::transform(counts.cbegin(), counts.cend(),
                   alignedSizes.begin(),
                   [](size_t s) -> Vector2ui
                   {
                       return Vector2ui(Memory::AlignSize((s) * sizeof(float)),
                                        Memory::AlignSize((s + 1) * sizeof(float)));
                   });
    Vector2ui totalSize = Zero2ui;
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {
        totalSize += alignedSizes[i];
    }

    // Allocate Memory
    // One for pdf and other for cdf
    size_t totalSizeLinear = totalSize[0] + totalSize[1];
    // Pointers
    memory = DeviceMemory(totalSizeLinear);
    size_t offset = 0;
    Byte* dPtr = static_cast<Byte*>(memory);
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {
        Vector2ui currentSize = alignedSizes[i];
        Byte* dPDFPtr = dPtr + offset;
        offset += currentSize[0];
        Byte* dCDFPtr = dPtr + offset;
        offset += currentSize[1];

        dPDFs.push_back(reinterpret_cast<const float*>(dPDFPtr));
        dCDFs.push_back(reinterpret_cast<const float*>(dCDFPtr));
    }
    assert(offset == totalSizeLinear);
}

void PWCDistributionGroupCPU1D::CopyPDFsConstructCDFs(const std::vector<const float*>& functionDataPtrs,
                                                         const CudaSystem& system,
                                                         cudaMemcpyKind copyKind)
{
    const CudaGPU& bestGPU = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(bestGPU.DeviceId()));
    for(size_t i = 0; i < counts.size(); i++)
    {
        CUDA_CHECK(cudaMemcpy(const_cast<float*>(dPDFs[i]),
                              functionDataPtrs[i],
                              counts[i] * sizeof(float),
                              copyKind));
        // Zero out the first cdf member
        //CUDA_CHECK(cudaMemset(const_cast<float*>(dCDFs[i]), 0x00, sizeof(float)));

        // Normalize PDF
        TransformArrayGPU(const_cast<float*>(dPDFs[i]), counts[i],
                          HostMultiplyFunctor<float>(1.0f / static_cast<float>(counts[i])));

        // Utilize GPU to do Scan Algorithm to find CDF
        ExclusiveScanArrayGPU<float, ReduceAdd<float>>(const_cast<float*>(dCDFs[i]),
                                                       dPDFs[i], counts[i] + 1, 0.0f);

        // Use last element to normalize Function values to PDF
        TransformArrayGPU(const_cast<float*>(dPDFs[i]), counts[i],
                          DeviceDivideFunctor<float>(dCDFs[i][counts[i]]));

        // Transform CDF Also
        TransformArrayGPU(const_cast<float*>(dCDFs[i]), counts[i] + 1,
                          DeviceDivideFunctor<float>(dCDFs[i][counts[i]]));
    }

    // Construct Objects
    for(size_t i = 0; i < counts.size(); i++)
    {
        gpuDistributions.push_back(PWCDistributionGPU1D(dCDFs[i],
                                                           dPDFs[i],
                                                           static_cast<uint32_t>(counts[i])));
    }
}

PWCDistributionGroupCPU1D::PWCDistributionGroupCPU1D(const std::vector<std::vector<float>>& functions,
                                                           const CudaSystem& system)
{
    // Gen Sizes
    counts.resize(functions.size());
    std::transform(functions.cbegin(), functions.cend(), counts.begin(),
                   [](const std::vector<float>& vec)
                   {
                       return vec.size();
                   });

    // Allocate and Generate Pointers for each function
    GeneratePointers();

    // Generate pointer array (since below function requires that)
    std::vector<const float*> functionDataPtrs;
    functionDataPtrs.reserve(functions.size());
    for(const auto& func : functions)
    {
        functionDataPtrs.push_back(func.data());
    }

    CopyPDFsConstructCDFs(functionDataPtrs, system, cudaMemcpyHostToDevice);
    // All Done!
}

PWCDistributionGroupCPU1D::PWCDistributionGroupCPU1D(const std::vector<const float*>& dFunctions,
                                                           const std::vector<size_t>& counts,
                                                           const CudaSystem& system)
{
    // Copy Sizes
    this->counts = counts;
    // Generate Pointers on the GPU
    GeneratePointers();
    CopyPDFsConstructCDFs(dFunctions, system, cudaMemcpyDeviceToDevice);
    // All Done!
}

const PWCDistributionGPU1D& PWCDistributionGroupCPU1D::DistributionGPU(uint32_t index) const
{
    return gpuDistributions[index];
}

const PWCDistributionGroupCPU1D::GPUDistList& PWCDistributionGroupCPU1D::DistributionGPU() const
{
    return gpuDistributions;
}

void PWCDistributionGroupCPU2D::Allocate(const std::vector<Vector2ui>& dimensions)
{
    this->dimensions = dimensions;
    std::vector<std::array<size_t, 5>> alignedSizes(dimensions.size());
    std::transform(dimensions.cbegin(), dimensions.cend(),
                   alignedSizes.begin(),
                   [](const Vector2ui& vec)->std::array<size_t, 5>
    {
        return
        {    // Row PDF Align Size
             Memory::AlignSize(vec[0] * sizeof(float)),
             // Row CDF Align Size
             Memory::AlignSize((vec[0] + 1) * sizeof(float)),
             // Column PDF Align Size
             Memory::AlignSize(vec[1] * sizeof(float)),
             // Column CDF Align Size
             Memory::AlignSize((vec[1] + 1) * sizeof(float)),
             // X Dist1D Align Size
             Memory::AlignSize(vec[1] * sizeof(PWCDistributionGPU1D))
        };
    });
    size_t totalSize = 0;
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {
        const auto& sizes = alignedSizes[i];
        const auto& dim = dimensions[i];

        totalSize += dim[1] * sizes[0];
        totalSize += dim[1] * sizes[1];
        totalSize += sizes[2];
        totalSize += sizes[3];
        totalSize += sizes[4];
    }

    // Allocate Memory
    memory = DeviceMemory(totalSize);

    size_t offset = 0;
    Byte* dPtr = static_cast<Byte*>(memory);
    std::vector<PWCDistributionGPU1D> hXDists;
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {
        DistData2D distData;
        const auto& sizes = alignedSizes[i];
        const Vector2ui& dimension = dimensions[i];

        hXDists.resize(dimension[1]);
        for(uint32_t y = 0; y < dimension[1]; y++)
        {
            distData.dXPDFs.push_back(reinterpret_cast<const float*>(dPtr + offset));
            offset += sizes[0];
            distData.dXCDFs.push_back(reinterpret_cast<const float*>(dPtr + offset));
            offset += sizes[1];
            hXDists[y] = PWCDistributionGPU1D(distData.dXCDFs.back(),
                                                 distData.dXPDFs.back(),
                                                 dimension[0]);
        }

        distData.dYPDF = reinterpret_cast<const float*>(dPtr + offset);
        offset += sizes[2];
        distData.dYCDF = reinterpret_cast<const float*>(dPtr + offset);
        offset += sizes[3];
        distData.dXDists = reinterpret_cast<const PWCDistributionGPU1D*>(dPtr + offset);
        offset += sizes[4];
        distData.yDist = PWCDistributionGPU1D(distData.dYCDF, distData.dYPDF,
                                                 dimension[1]);

        // Memcpy Constructed 1D Distributions
        CUDA_CHECK(cudaMemcpy(const_cast<PWCDistributionGPU1D*>(distData.dXDists),
                              hXDists.data(), dimension[1] * sizeof(PWCDistributionGPU1D),
                              cudaMemcpyHostToDevice));

        distDataList.push_back(distData);
    }
    assert(offset == totalSize);

    for(size_t i = 0; i < alignedSizes.size(); i++)
    {
        const DistData2D& distData = distDataList[i];
        // Construct 2D Dist
        PWCDistributionGPU2D dist(distData.yDist, distData.dXDists,
                                     dimensions[i][0], dimensions[i][1]);
        gpuDistributions.push_back(dist);
    }
}

PWCDistributionGroupCPU2D::PWCDistributionGroupCPU2D(const std::vector<const float*>& dFunctions,
                                                           const std::vector<Vector2ui>& dimensions,
                                                           const std::vector<bool>& factorSpherical,
                                                           const CudaSystem& system)
{
    // Allocate
    Allocate(dimensions);
    UpdateDistributions(dFunctions, factorSpherical,
                        system, cudaMemcpyHostToDevice);
}

PWCDistributionGroupCPU2D::PWCDistributionGroupCPU2D(const std::vector<std::vector<float>>& functionValues,
                                                           const std::vector<Vector2ui>& dimensions,
                                                           const std::vector<bool>& factorSpherical,
                                                           const CudaSystem& system)
    : dimensions(dimensions)
{

    std::vector<const float*> functionDataPtrs;
    functionDataPtrs.reserve(functionValues.size());
    for(const auto& funcVector : functionValues)
    {
        functionDataPtrs.push_back(funcVector.data());
    }

    Allocate(dimensions);
    UpdateDistributions(functionDataPtrs, factorSpherical,
                        system, cudaMemcpyHostToDevice);
}

const PWCDistributionGPU2D& PWCDistributionGroupCPU2D::DistributionGPU(uint32_t index) const
{
    return gpuDistributions[index];
}

const PWCDistributionGroupCPU2D::GPUDistList& PWCDistributionGroupCPU2D::DistributionGPU() const
{
    return gpuDistributions;
}

void PWCDistributionGroupCPU2D::UpdateDistributions(const std::vector<const float*>& functionDataPtrs,
                                                       const std::vector<bool>& factorSpherical,
                                                       const CudaSystem& system, cudaMemcpyKind kind)
{
    CUDA_CHECK(cudaSetDevice(system.BestGPU().DeviceId()));
    const CudaGPU& gpu = system.BestGPU();
    // Generate CDFs for each distributions &
    // Construct 2D Distributions
    for(size_t i = 0; i < functionDataPtrs.size(); i++)
    {
        const DistData2D& distData = distDataList[i];
        const Vector2ui& dim = dimensions[i];
        bool factorSphr = factorSpherical[i];

        for(uint32_t y = 0; y < dim[1]; y++)
        {
            const float* rowFunctionValues = functionDataPtrs[i] + (y * dim[0]);
            float& rowFunction = const_cast<float&>(distData.dYPDF[y]);

            float* dRowPDF = const_cast<float*>(distData.dXPDFs[y]);
            float* dRowCDF = const_cast<float*>(distData.dXCDFs[y]);

            CUDA_CHECK(cudaMemcpyAsync(dRowPDF, rowFunctionValues,
                                  dim[0] * sizeof(float),
                                  kind));

            // From PBR-Book factoring in the spherical phi term
            if(factorSphr)
            {
                float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(dim[1]);
                // V is [0,1] convert it to [0,pi]
                // so that middle sections will have higher priority
                float phi = v  * MathConstants::Pi;
                float sinPhi = std::sin(phi);
                HostMultiplyFunctor<float> sphericalTermFunctor(sinPhi);
                TransformArrayGPU(dRowPDF, dim[0], sphericalTermFunctor);
            }

            // Currently dRowPDF is the function value
            // Reduce it first to get row weight
            ReduceArrayGPU<float, ReduceAdd<float>>(rowFunction, dRowPDF, dim[0], 0.0f);

            // Normalize PDF
            //HostMultiplyFunctor<float> normLength(1.0f / static_cast<float>(dim[0]));
            //HostMultiplyFunctor<float> length(static_cast<float>(dim[0]));
            TransformArrayGPU(dRowPDF, dim[0],
                              HostMultiplyFunctor<float>(1.0f / static_cast<float>(dim[0])));

            // Utilize GPU to do Scan Algorithm to find CDF
            ExclusiveScanArrayGPU<float, ReduceAdd<float>>(dRowCDF, dRowPDF,
                                                           dim[0] + 1, 0.0f);

            // Use last element to normalize Function values to PDF
            // Since we used PDF buffer to calculate CDF
            // multiply the function with the size as well
            DeviceHostMulDivideComboFunctor<float> pdfNormFunctor(dRowCDF[dim[0]], static_cast<float>(dim[0]));
            TransformArrayGPU(dRowPDF, dim[0], pdfNormFunctor);
            // Normalize CDF with the total accumulation (last element)
            // to perfectly match the [0,1) interval
            DeviceDivideFunctor<float> cdfNormFunctor(dRowCDF[dim[0]]);
            TransformArrayGPU(dRowCDF, dim[0] + 1, cdfNormFunctor);
        }

        float* dYPDF = const_cast<float*>(distData.dYPDF);
        float* dYCDF = const_cast<float*>(distData.dYCDF);

        // Generate Y Axis Distribution Data
        TransformArrayGPU(dYPDF, dim[1],
                          HostMultiplyFunctor<float>(1.0f / static_cast<float>(dim[1])));

        ExclusiveScanArrayGPU<float, ReduceAdd<float>>(dYCDF, dYPDF,
                                                       dim[1] + 1, 0.0f);
        // Use last element to normalize Function values to PDF
        // Since we used PDF buffer to calculate CDF
        // multiply the function with the size as well
        DeviceDivideFunctor<float> cdfNormFunctor(dYCDF[dim[1]]);
        DeviceHostMulDivideComboFunctor<float> pdfNormFunctor(dYCDF[dim[1]], static_cast<float>(dim[1]));
        TransformArrayGPU(dYPDF, dim[1], pdfNormFunctor);
        // Normalize CDF with the total accumulation (last element)
        // to perfectly match the [0,1) interval
        TransformArrayGPU(dYCDF, dim[1] + 1, cdfNormFunctor);
    }
    // All Done!
    gpu.WaitMainStream();

    // DEBUG WRITE TO FILE
    //uint32_t i = 0;
    //for(const auto& dist : gpuDistributions)
    //{
    //    Vector2ui dim = dimensions[i];
    //    std::string name = std::to_string(i) + "_dist";

    //    Debug::DumpMemToFile(name, dist.gDistributionY.gPDF, dim[1]);
    //    Debug::DumpMemToFile(name, dist.gDistributionY.gCDF, dim[1] + 1, true);

    //    for(uint32_t j = 0; j < dim[1]; j++)
    //    {
    //        Debug::DumpMemToFile(name, dist.gDistributionsX[j].gPDF, dim[0], true);
    //        Debug::DumpMemToFile(name, dist.gDistributionsX[j].gCDF, dim[0] + 1, true);
    //    }
    //    i++;
    //}
}

void PWCDistStaticCPU2D::Allocate(uint32_t distCount,
                                   const Vector2ui& dimensions)
{
    // Allocate Required Data
    uint32_t dataPerDist = dimensions.Multiply();
    uint32_t yPDFCount = dimensions[1] * distCount;
    uint32_t yCDFCount = (dimensions[1] + 1) * distCount;
    uint32_t xPDFCount = dataPerDist * distCount;
    uint32_t xCDFCount = (dataPerDist + dimensions[0]) * distCount;

    GPUMemFuncs::AllocateMultiData(std::tie(gpuDist.gXPDFs,
                                            gpuDist.gXCDFs,
                                            gpuDist.gYPDFs,
                                            gpuDist.gYCDFs),
                                   memory,
                                   {xPDFCount, xCDFCount,
                                    yPDFCount, yCDFCount});

    gpuDist.dim = dimensions;
    gpuDist.distCount = distCount;
}

PWCDistStaticCPU2D::PWCDistStaticCPU2D(const float* dFunctions,
                                         uint32_t distCount,
                                         Vector2ui dimensions,
                                         bool factorSpherical,
                                         const CudaSystem& system)
{
    Allocate(distCount, dimensions);
    UpdateDistributions(dFunctions, factorSpherical, system,
                        cudaMemcpyDeviceToDevice);
}

PWCDistStaticCPU2D::PWCDistStaticCPU2D(std::vector<float>& functions,
                                         uint32_t distCount,
                                         Vector2ui dimensions,
                                         bool factorSpherical,
                                         const CudaSystem& system)
{
    Allocate(distCount, dimensions);
    UpdateDistributions(functions.data(), factorSpherical, system,
                        cudaMemcpyHostToDevice);
}

PWCDistStaticGPU2D PWCDistStaticCPU2D::DistributionGPU() const
{
    return gpuDist;
}

void PWCDistStaticCPU2D::UpdateDistributions(const float* functionData,
                                              bool factorSpherical,
                                              const CudaSystem& system,
                                              cudaMemcpyKind kind)
{
    // Copy the functions to the GPU
    uint32_t totalCount = gpuDist.dim.Multiply() * gpuDist.distCount;
    CUDA_CHECK(cudaMemcpy(const_cast<float*>(gpuDist.gXPDFs), functionData,
                          totalCount * sizeof(float),
                          kind));

    // Call a single kernel
    // 1 block per distribution
    const CudaGPU& gpu = system.BestGPU();

    gpu.ExactKC_X(0, (cudaStream_t)0,
                  StaticThreadPerBlock1D, gpuDist.distCount,
                  //
                  KCUpdateDistributionsX,
                  //
                  const_cast<float*>(gpuDist.gXPDFs),
                  const_cast<float*>(gpuDist.gXCDFs),
                  const_cast<float*>(gpuDist.gYPDFs),
                  gpuDist.dim,
                  gpuDist.distCount,
                  factorSpherical);

    // Now do the Y
    gpu.ExactKC_X(0, (cudaStream_t)0,
                  StaticThreadPerBlock1D, gpuDist.distCount,
                  //
                  KCUpdateDistributionsY,
                  //
                  const_cast<float*>(gpuDist.gYPDFs),
                  const_cast<float*>(gpuDist.gYCDFs),
                  gpuDist.dim,
                  gpuDist.distCount);


    //uint32_t yPDFCount = gpuDist.dim[1] * gpuDist.distCount;
    //uint32_t yCDFCount = (gpuDist.dim[1] + 1) * gpuDist.distCount;
    //uint32_t xPDFCount = gpuDist.dim.Multiply() * gpuDist.distCount;
    //uint32_t xCDFCount = (gpuDist.dim.Multiply() + gpuDist.dim[0]) * gpuDist.distCount;

    //Debug::DumpMemToFile("yPDFs", gpuDist.gYPDFs, yPDFCount);
    //Debug::DumpMemToFile("yCDFs", gpuDist.gYCDFs, yCDFCount);
    //Debug::DumpMemToFile("xPDFs", gpuDist.gXPDFs, xPDFCount);
    //Debug::DumpMemToFile("xCDFs", gpuDist.gXCDFs, xCDFCount);
}