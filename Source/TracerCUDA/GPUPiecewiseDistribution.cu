#include "GPUPiecewiseDistribution.cuh"
#include "ParallelScan.cuh"

#include "RayLib/Types.h"
#include "RayLib/MemoryAlignment.h"

#include <numeric>

CPUDistGroupPiecewise1D::CPUDistGroupPiecewise1D(const std::vector<std::vector<float>>& pdfValues,
                                                 const CudaSystem& system)
{
    const CudaGPU& bestGPU = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(bestGPU.DeviceId()));

    // Gen Sizes
    counts.resize(pdfValues.size());    
    std::transform(pdfValues.cbegin(), pdfValues.cend(), counts.begin(),
                   [](const std::vector<float>& vec)
                   {
                       return vec.size();
                   });

    std::vector<size_t> alignedSizes(counts.size());
    std::transform(alignedSizes.cbegin(), alignedSizes.cend(),
                   alignedSizes.begin(),
                   [](size_t s)
                   {
                       return Memory::AlignSize(s * sizeof(float));
                   });
    size_t totalSize = std::reduce(alignedSizes.cbegin(), alignedSizes.cend());

    // Allocate Memory
    // One for pdf and other for cdf
    totalSize = totalSize * 2;
    // Pointers
    memory = DeviceMemory(totalSize);
    size_t offset = 0;
    Byte* dPtr = static_cast<Byte*>(memory);
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {
        size_t currentSize = alignedSizes[i];
        Byte* dPDFPtr = dPtr + offset;
        offset += currentSize;
        Byte* dCDFPtr = dPtr + offset;
        offset += currentSize;

        dPDFs.push_back(reinterpret_cast<float*>(dPDFPtr));
        dCDFs.push_back(reinterpret_cast<float*>(dCDFPtr));
    }
    assert(offset == totalSize);

    // Construct CDFs and Memcpy
    std::vector<float> cdfValues;
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {        
        CUDA_CHECK(cudaMemcpy(const_cast<float*>(dPDFs[i]), 
                              pdfValues[i].data(),                              
                              counts[i] * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        // Utilize GPU to do Scan Algorithm to find CDF
        ExclusiveScanArrayGPU<float, []()(dCDFs[i], dPDFs[i], counts[i], 0u);
        
        std::exclusive_scan(pdfs.cbegin(), pdfs.cend(),
                            cdfValues.begin(), 0);
    }

    // Construct Objects
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {
        gpuDistributions.push_back(GPUDistPiecewise1D(dCDFs[i],
                                                      dPDFs[i],
                                                      counts[i]));
    }
    // All Done!
}

const GPUDistPiecewise1D& CPUDistGroupPiecewise1D::DistributionGPU(uint32_t index) const
{
    return gpuDistributions[index];
}

CPUDistGroupPiecewise2D::CPUDistGroupPiecewise2D(const std::vector<std::vector<float>>& pdfValues,
                                                 const std::vector<Vector2ui>& dimensions,
                                                 const CudaSystem& system)
    : dimensions(dimensions)
{
    // Gen Sizes
    std::vector<Vector2ui> counts(dimensions.size());
    std::transform(dimensions.cbegin(), dimensions.cend(), counts.begin(),
                   [](const Vector2ui& vec)
                   {
                       // PDF Count values for X and Y dimensions
                       return Vector2ui(vec[0] * vec[1],
                                        vec[1]);
                   });

    std::vector<Vector2ui> alignedSizes(counts.size());
    std::transform(alignedSizes.cbegin(), alignedSizes.cend(),
                   alignedSizes.begin(),
                   [](const Vector2ui& vec)
                   {
                       return Vector2ui(Memory::AlignSize(vec[0] * sizeof(float)),
                                        Memory::AlignSize(vec[1] * sizeof(float)));
                   });
    Vector2ui dataTotalSize = std::reduce(alignedSizes.cbegin(), alignedSizes.cend());
    // Calculate X Distribution sizes
    std::vector<uint32_t> xDistAlignedSize(dimensions.size());
    std::transform(dimensions.cbegin(), dimensions.cend(), xDistAlignedSize.begin(),
                   [](const Vector2ui& vec)
                   {
                       return Memory::AlignSize(vec[0] * sizeof(GPUDistPiecewise1D));
                   });
    size_t xDistTotalSize = std::reduce(xDistAlignedSize.cbegin(), xDistAlignedSize.cend());
    
    // Allocate Memory
    size_t totalSize = (xDistTotalSize +
                        dataTotalSize[0] * 2 + 
                        dataTotalSize[1] * 2);
    memory = DeviceMemory(totalSize);



}

const GPUDistPiecewise2D& CPUDistGroupPiecewise2D::DistributionGPU(uint32_t index) const
{
    return gpuDistributions[index];
}

