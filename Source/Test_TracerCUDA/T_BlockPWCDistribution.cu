#include "TracerCUDA/GPUBlockPWCDistribution.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DeviceMemory.h"

#include "RayLib/Constants.h"

#include <gtest/gtest.h>
#include <random>
#include <numeric>

static constexpr uint32_t TPB = 32;
static constexpr uint32_t X = 32;
static constexpr uint32_t Y = 1;
static constexpr uint32_t PIX_COUNT = X * Y;
static constexpr uint32_t DATA_PER_BLOCK = PIX_COUNT / TPB;

static constexpr uint32_t X_CDF_COUNT = (X + 1) * Y;
static constexpr uint32_t Y_CDF_COUNT = (Y + 1);

__global__ __launch_bounds__(TPB)
void KCPiecewiseConstDistInitCheck(float* gPDFXOut,
                                   float* gCDFXOut,
                                   float* gPDFYOut,
                                   float* gCDFYOut,
                                   const float* gData)
{
    const uint32_t threadId = threadIdx.x;

    using BlockPWC2D = BlockPWCDistribution2D<TPB, X, Y>;
    // Allocate shared memory for Block Operations
    __shared__ typename BlockPWC2D::TempStorage sPWCMem;

    float data[DATA_PER_BLOCK];
    for(uint32_t i = 0; i < DATA_PER_BLOCK; i++)
    {
        data[i] = gData[i * TPB + threadId];
    }

    BlockPWC2D dist2D(sPWCMem, data);

    dist2D.DumpSharedMem(gPDFXOut,
                         gCDFXOut,
                         gPDFYOut,
                         gCDFYOut);

}

TEST(BlockPWCDistribution2D, BasicInit)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    // Copy all ones to GPU
    std::vector<float> data(PIX_COUNT, 1.0f);

    // GPU Allocations
    float* dData;
    float* dPDFX;
    float* dCDFX;
    float* dPDFY;
    float* dCDFY;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dData,
                                            dPDFX, dCDFX,
                                            dPDFY, dCDFY),
                                   mem,
                                   {PIX_COUNT,
                                    PIX_COUNT, X_CDF_COUNT,
                                    Y, Y_CDF_COUNT});
    CUDA_CHECK(cudaMemcpy(dData, data.data(), sizeof(float) * PIX_COUNT,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dPDFX, 0xFF, sizeof(float) * PIX_COUNT));
    CUDA_CHECK(cudaMemset(dCDFX, 0xFF, sizeof(float) * X_CDF_COUNT));
    CUDA_CHECK(cudaMemset(dPDFY, 0xFF, sizeof(float) * Y));
    CUDA_CHECK(cudaMemset(dCDFY, 0xFF, sizeof(float) * Y_CDF_COUNT));

    // Scan Call
    const CudaGPU& bestGPU = system.BestGPU();
    bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                      //
                      KCPiecewiseConstDistInitCheck,
                      //
                      dPDFX,
                      dCDFX,
                      dPDFY,
                      dCDFY,
                      dData);

    // Copy to Host to check
    std::vector<float> hPDFX(PIX_COUNT);
    std::vector<float> hCDFX(X_CDF_COUNT);
    std::vector<float> hPDFY(Y);
    std::vector<float> hCDFY(Y_CDF_COUNT);
    CUDA_CHECK(cudaMemcpy(hPDFX.data(), dPDFX, sizeof(float) * PIX_COUNT,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCDFX.data(), dCDFX, sizeof(float) * X_CDF_COUNT,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hPDFY.data(), dPDFY, sizeof(float) * Y,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCDFY.data(), dCDFY, sizeof(float) * Y_CDF_COUNT,
                          cudaMemcpyDeviceToHost));
    // Checks
    for(float pdf : hPDFX)
    {
        EXPECT_FLOAT_EQ(pdf, 1.0f);
    }
    for(uint32_t i = 0; i < X_CDF_COUNT; i++)
    {

    }
}
