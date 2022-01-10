#include "GPUPiecewiseDistribution.cuh"
#include "ParallelScan.cuh"
#include "ParallelReduction.cuh"
#include "ParallelTransform.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

#include "RayLib/Types.h"
#include "RayLib/MemoryAlignment.h"

#include <numeric>

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
__device__  __forceinline__
T DeviceHostMulDivideComboFunctor<T>::operator()(const T& in) const
{
    return in * mulValue / gDivValue;
}

void CPUDistGroupPiecewiseConst1D::GeneratePointers()
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

void CPUDistGroupPiecewiseConst1D::CopyPDFsConstructCDFs(const std::vector<const float*>& functionDataPtrs,
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
        TransformArrayGPU(const_cast<float*>(dCDFs[i]), counts[i],
                          DeviceDivideFunctor<float>(dCDFs[i][counts[i]]));
    }

    // Construct Objects
    for(size_t i = 0; i < counts.size(); i++)
    {
        gpuDistributions.push_back(GPUDistPiecewiseConst1D(dCDFs[i],
                                                           dPDFs[i],
                                                           static_cast<uint32_t>(counts[i])));
    }
}

CPUDistGroupPiecewiseConst1D::CPUDistGroupPiecewiseConst1D(const std::vector<std::vector<float>>& functions,
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
    std::vector<const float*> functionDataPtrs(functions.size());
    for(const auto& func : functions)
    {
        functionDataPtrs.push_back(func.data());
    }

    CopyPDFsConstructCDFs(functionDataPtrs, system, cudaMemcpyHostToDevice);
    // All Done!
}

CPUDistGroupPiecewiseConst1D::CPUDistGroupPiecewiseConst1D(const std::vector<const float*>& dFunctions,
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

const GPUDistPiecewiseConst1D& CPUDistGroupPiecewiseConst1D::DistributionGPU(uint32_t index) const
{
    return gpuDistributions[index];
}

const CPUDistGroupPiecewiseConst1D::GPUDistList& CPUDistGroupPiecewiseConst1D::DistributionGPU() const
{
    return gpuDistributions;
}

CPUDistGroupPiecewiseConst2D::CPUDistGroupPiecewiseConst2D(const std::vector<std::vector<float>>& functionValues,
                                                           const std::vector<Vector2ui>& dimensions,
                                                           const std::vector<bool>& factorSpherical,
                                                           const CudaSystem& system)
    : dimensions(dimensions)
{
    CUDA_CHECK(cudaSetDevice(system.BestGPU().DeviceId()));

    std::vector<std::array<size_t, 5>> alignedSizes(dimensions.size());
    std::transform(dimensions.cbegin(), dimensions.cend(),
                   alignedSizes.begin(),
                   [](const Vector2ui& vec) -> std::array<size_t, 5>
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
            hXDists[y] = GPUDistPiecewiseConst1D(distData.dXCDFs.back(),
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
        bool factorSphr = factorSpherical[i];

        //// Yolo check image
        //const float* pixels = functionValues[i].data();
        //float pix0, pix1;
        //pix0 = pixels[dim[0] * 176 + 2456];
        //pix1 = pixels[dim[0] * (2048 - 176) + 2456];
        //METU_LOG("Pix on Dist (%f) (%f)", pix0, pix1);

        for(uint32_t y = 0; y < dim[1]; y++)
        {
            const float* rowFunctionValues = functionValues[i].data() + (y * dim[0]);
            float& rowFunction = const_cast<float&>(distData.dYPDF[y]);

            float* dRowPDF = const_cast<float*>(distData.dXPDFs[y]);
            float* dRowCDF = const_cast<float*>(distData.dXCDFs[y]);

            CUDA_CHECK(cudaMemcpy(dRowPDF, rowFunctionValues,
                                  dim[0] * sizeof(float),
                                  cudaMemcpyHostToDevice));

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
            HostMultiplyFunctor<float> normLength(1.0f / static_cast<float>(dim[0]));
            HostMultiplyFunctor<float> length(static_cast<float>(dim[0]));
            TransformArrayGPU(dRowPDF, dim[0],
                              HostMultiplyFunctor<float>(1.0f / static_cast<float>(dim[0])));

            // Utilize GPU to do Scan Algorithm to find CDF
            ExclusiveScanArrayGPU<float, ReduceAdd<float>>(dRowCDF, dRowPDF,
                                                           dim[0] + 1u,
                                                           0.0f);

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

        ExclusiveScanArrayGPU<float, ReduceAdd<float>>(dYCDF, dYPDF, (dim[1] + 1), 0.0f);
        // Use last element to normalize Function values to PDF
        // Since we used PDF buffer to calculate CDF
        // multiply the function with the size as well
        DeviceDivideFunctor<float> cdfNormFunctor(dYCDF[dim[1]]);
        DeviceHostMulDivideComboFunctor<float> pdfNormFunctor(dYCDF[dim[1]], static_cast<float>(dim[1]));
        TransformArrayGPU(dYPDF, dim[1], pdfNormFunctor);
        // Normalize CDF with the total accumulation (last element)
        // to perfectly match the [0,1) interval
        TransformArrayGPU(dYCDF, dim[1] + 1, cdfNormFunctor);

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

const CPUDistGroupPiecewiseConst2D::GPUDistList& CPUDistGroupPiecewiseConst2D::DistributionGPU() const
{
    return gpuDistributions;
}