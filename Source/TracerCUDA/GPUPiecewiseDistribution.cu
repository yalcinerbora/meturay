#include "GPUPiecewiseDistribution.cuh"
#include "ParallelScan.cuh"
#include "ParallelReduction.cuh"
#include "ParallelTransform.cuh"

#include "RayLib/Types.h"
#include "RayLib/MemoryAlignment.h"

#include <numeric>

CPUDistGroupPiecewiseConst1D::CPUDistGroupPiecewiseConst1D(const std::vector<std::vector<float>>& functions,
                                                           const CudaSystem& system)
{
    const CudaGPU& bestGPU = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(bestGPU.DeviceId()));

    // Gen Sizes
    counts.resize(functions.size());
    std::transform(functions.cbegin(), functions.cend(), counts.begin(),
                   [](const std::vector<float>& vec)
                   {
                       return vec.size();
                   });

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

    // Construct CDFs and Memcpy
    std::vector<float> cdfValues;
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {        
        CUDA_CHECK(cudaMemcpy(const_cast<float*>(dPDFs[i]), 
                              functions[i].data(),
                              counts[i] * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        // Normalize PDF
        TransformArrayGPU(dPDFs[i], counts[i],
                          HostMultiplyFunctor<float>(1.0f / static_cast<float>(counts[i])));

        // Utilize GPU to do Scan Algorithm to find CDF
        ExclusiveScanArrayGPU<float, ReduceAdd<float>>(const_cast<float*>(dCDFs[i]),
                                                       dPDFs[i], counts[i] + 1, 0.0f);

        // Use last element to normalize Function values to PDF
        TransformArrayGPU(dPDFs[i], counts[i],
                          DeviceDivideFunctor<float>(dCDFs[i][counts[i]]));

        // Transform CDF Also
        TransformArrayGPU(dCDFs[i], counts[i],
                          DeviceDivideFunctor<float>(dCDFs[i][counts[i]]));
    }

    // Construct Objects
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {
        gpuDistributions.push_back(GPUDistPiecewiseConst1D(dCDFs[i],
                                                           dPDFs[i],
                                                           static_cast<uint32_t>(counts[i])));
    }
    // All Done!
}

const GPUDistPiecewiseConst1D& CPUDistGroupPiecewiseConst1D::DistributionGPU(uint32_t index) const
{
    return gpuDistributions[index];
}

CPUDistGroupPiecewiseConst2D::CPUDistGroupPiecewiseConst2D(const std::vector<std::vector<float>>& functionValues,
                                                           const std::vector<Vector2ui>& dimensions,
                                                           const CudaSystem& system)
    : dimensions(dimensions)
{
    std::vector<std::array<size_t, 5>> alignedSizes(dimensions.size());
    std::transform(dimensions.cbegin(), dimensions.cend(),
                   alignedSizes.begin(),
                   [](const Vector2ui& vec) -> std::array<size_t, 5>
                   {
                       return {// Row PDF Align Size
                                        Memory::AlignSize(vec[0] * sizeof(float)),
                                        // Row CDF Align Size
                                        Memory::AlignSize((vec[0] + 1) * sizeof(float)),
                                        // Column PDF Align Size
                                        Memory::AlignSize(vec[1] * sizeof(float)),
                                        // Column CDF Align Size
                                        Memory::AlignSize((vec[1] + 1) * sizeof(float)),
                                        // X Dist1D Align Size
                                        Memory::AlignSize(vec[1] * sizeof(GPUDistPiecewiseConst1D))
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
    std::vector<GPUDistPiecewiseConst1D> hXDists;
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
            hXDists[i] = GPUDistPiecewiseConst1D(distData.dXCDFs.back(),
                                                 distData.dXPDFs.back(),
                                                 dimension[0]);
        }

        distData.dYPDF = reinterpret_cast<const float*>(dPtr + offset);
        offset += sizes[2];
        distData.dYCDF = reinterpret_cast<const float*>(dPtr + offset);
        offset += sizes[3];
        distData.dXDists = reinterpret_cast<const GPUDistPiecewiseConst1D*>(dPtr + offset);
        offset += sizes[4];
        distData.yDist = GPUDistPiecewiseConst1D(distData.dYCDF, distData.dYPDF,
                                                 dimension[1]);

        // Memcpy Constructed 1D Distributions
        CUDA_CHECK(cudaMemcpy(const_cast<GPUDistPiecewiseConst1D*>(distData.dXDists), 
                              hXDists.data(), dimension[1] * sizeof(GPUDistPiecewiseConst1D),
                              cudaMemcpyHostToDevice));

        distDataList.push_back(distData);
    }
    assert(offset == totalSize);

    // Generate CDFs for each distributions &
    // Construct 2D Distributions
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {
        const DistData2D& distData = distDataList[i];
        const Vector2ui& dim = dimensions[i];

        for(uint32_t y = 0; y < dim[1]; y++)
        {
            const float* rowFunctionValues = functionValues[i].data() + (y * dim[1]);
            float& rowFunction = const_cast<float&>(distData.dYPDF[y]);
           
            float* dRowPDF = const_cast<float*>(distData.dXPDFs[i]);
            float* dRowCDF = const_cast<float*>(distData.dXPDFs[i]);

            CUDA_CHECK(cudaMemcpy(dRowPDF, rowFunctionValues, 
                                  dim[0] * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // Normalize PDF
            TransformArrayGPU(dRowPDF, dim[0],
                              HostMultiplyFunctor<float>(1.0f / static_cast<float>(dim[0])));

            // Utilize GPU to do Scan Algorithm to find CDF
            ExclusiveScanArrayGPU<float, ReduceAdd<float>>(dRowCDF, dRowPDF,
                                                           dim[i] + 1, 
                                                           0.0f);

            // Use last element to normalize Function values to PDF
            DeviceDivideFunctor<float> normlizeFunctor(dRowCDF[dim[0]]);
            TransformArrayGPU(dRowPDF, dim[0], normlizeFunctor);
            // Normalize CDF Also
            TransformArrayGPU(dRowCDF, dim[0] + 1, normlizeFunctor);   

            // Reduce PDFs for Y dim
            ReduceArrayGPU<float, ReduceAdd<float>>(rowFunction, dRowPDF, dim[0], 0.0f);
        }

        float* dYPDF = const_cast<float*>(distData.dYPDF);
        float* dYCDF = const_cast<float*>(distData.dYCDF);

        // Generate Y Axis Distribution Data
        TransformArrayGPU(dYPDF, dim[1],
                          HostMultiplyFunctor<float>(1.0f / static_cast<float>(dim[1])));

        ExclusiveScanArrayGPU<float, ReduceAdd<float>>(dYCDF, dYPDF, (dim[1] + 1), 0.0f);
        // Use last element to normalize Function values to PDF
        DeviceDivideFunctor<float> normlizeFunctor(dYCDF[dim[1]]);
        TransformArrayGPU(dYPDF, dim[1], normlizeFunctor);
        // Normalize CDF Also
        TransformArrayGPU(dYCDF, dim[1] + 1, normlizeFunctor);
        
        // Construct 2D Dist
        GPUDistPiecewiseConst2D dist(distData.yDist, distData.dXDists,
                                     dimensions[i][0], dimensions[i][1]);
        gpuDistributions.push_back(dist);
    }   
    // All Done!
}

const GPUDistPiecewiseConst2D& CPUDistGroupPiecewiseConst2D::DistributionGPU(uint32_t index) const
{
    return gpuDistributions[index];
}

