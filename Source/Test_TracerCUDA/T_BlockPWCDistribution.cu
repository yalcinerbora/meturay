#include "TracerCUDA/GPUBlockPWCDistribution.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DeviceMemory.h"

#include "RayLib/Constants.h"

#include <gtest/gtest.h>
#include <random>
#include <numeric>

static constexpr uint32_t TPB = 512;
static constexpr uint32_t X = 32;
static constexpr uint32_t Y = 32;
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

    // PWC Initialization and Dump to Global Memory Call
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

    for(uint32_t j = 0; j < Y; j++)
    for(uint32_t i = 0; i < (X + 1); i++)
    {
        static constexpr float DELTA_X = 1.0f / static_cast<float>(X);

        uint32_t index = j * (X + 1) + i;
        float value = static_cast<float>(i) * DELTA_X;
        EXPECT_FLOAT_EQ(value, hCDFX[index]);
    }
    // Marginal
    for(float pdf : hPDFY)
    {
        EXPECT_FLOAT_EQ(pdf, 1.0f);
    }
    for(uint32_t i = 0; i < Y_CDF_COUNT; i++)
    {
        static constexpr float DELTA_Y = 1.0f / static_cast<float>(Y);
        float value = static_cast<float>(i) * DELTA_Y;
        EXPECT_FLOAT_EQ(value, hCDFY[i]);
    }
}

TEST(BlockPWCDistribution2D, InitStree)
{
    static constexpr uint32_t ITERATION_COUNT = 100;

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    // CPU Allocations
    std::vector<float> hData(PIX_COUNT);
    std::vector<float> hPDFX(PIX_COUNT);
    std::vector<float> hCDFX(X_CDF_COUNT);
    std::vector<float> hPDFY(Y);
    std::vector<float> hCDFY(Y_CDF_COUNT);
    std::vector<float> hPDFXExpected(PIX_COUNT);
    std::vector<float> hCDFXExpected(X_CDF_COUNT);
    std::vector<float> hPDFYExpected(Y);
    std::vector<float> hCDFYExpected(Y_CDF_COUNT);
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

    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<float> uniformDist(0.0f, 10.0f);

    for(uint32_t i = 0; i < ITERATION_COUNT; i++)
    {
        // Generate new batch of random numbers
        for(float& d : hData)
        {
            d = uniformDist(rng);
        }

        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(dData, hData.data(), sizeof(float) * PIX_COUNT,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dPDFX, 0xFF, sizeof(float) * PIX_COUNT));
        CUDA_CHECK(cudaMemset(dCDFX, 0xFF, sizeof(float) * X_CDF_COUNT));
        CUDA_CHECK(cudaMemset(dPDFY, 0xFF, sizeof(float) * Y));
        CUDA_CHECK(cudaMemset(dCDFY, 0xFF, sizeof(float) * Y_CDF_COUNT));

        // PWC Initialization and Dump to Global Memory Call
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
        CUDA_CHECK(cudaMemcpy(hPDFX.data(), dPDFX, sizeof(float) * PIX_COUNT,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hCDFX.data(), dCDFX, sizeof(float) * X_CDF_COUNT,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hPDFY.data(), dPDFY, sizeof(float) * Y,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hCDFY.data(), dCDFY, sizeof(float) * Y_CDF_COUNT,
                              cudaMemcpyDeviceToHost));

        // Generate Marginal and Conditional Data here
        for(uint32_t j = 0; j < Y; j++)
        {
            static constexpr float DELTA_X = 1.0f / static_cast<float>(X);
            const uint32_t dataRowStart = j * X;
            const uint32_t dataRowEnd = (j + 1) * X;

            std::transform(hData.cbegin() + dataRowStart,
                           hData.cbegin() + dataRowEnd,
                           hPDFXExpected.begin() + dataRowStart,
                           [&](float f)
                           {
                               return f * DELTA_X;
                           });
            float totalSum = std::reduce(hPDFXExpected.cbegin() + dataRowStart,
                                         hPDFXExpected.cbegin() + dataRowEnd);
            hPDFYExpected[j] = totalSum;

            uint32_t cdfRowStart = j * (X + 1);
            uint32_t cdfRowEnd = (j + 1) * (X + 1);
            std::inclusive_scan(hPDFXExpected.cbegin() + dataRowStart,
                                hPDFXExpected.cbegin() + dataRowEnd,
                                hCDFXExpected.begin() + (cdfRowStart + 1));
            hCDFXExpected[cdfRowStart] = 0.0f;
            if(totalSum != 0.0f)
            {
                std::transform(hPDFXExpected.cbegin() + dataRowStart,
                               hPDFXExpected.cbegin() + dataRowEnd,
                               hPDFXExpected.begin() + dataRowStart,
                               [&](float f)
                               {
                                   return f / totalSum;
                               });
                std::transform(hCDFXExpected.cbegin() + cdfRowStart,
                               hCDFXExpected.cbegin() + cdfRowEnd,
                               hCDFXExpected.begin() + cdfRowStart,
                               [&](float f)
                               {
                                   return f / totalSum;
                               });
            }
            // Transform the PDF Back
            std::transform(hPDFXExpected.cbegin() + dataRowStart,
                           hPDFXExpected.cbegin() + dataRowEnd,
                           hPDFXExpected.begin() + dataRowStart,
                           [&](float f)
                           {
                               return f * static_cast<float>(X);
                           });
        }


        static constexpr float DELTA_Y = 1.0f / static_cast<float>(Y);
        std::transform(hPDFYExpected.cbegin(), hPDFYExpected.cend(),
                       hPDFYExpected.begin(),
                       [&](float f)
                       {
                           return f * DELTA_Y;
                       });
        std::inclusive_scan(hPDFYExpected.cbegin(), hPDFYExpected.cend(),
                            hCDFYExpected.begin() + 1);
        hCDFYExpected[0] = 0.0f;
        float totalSum = hCDFYExpected.back();

        if(totalSum != 0.0f)
        {
            std::transform(hPDFYExpected.cbegin(), hPDFYExpected.cend(),
                           hPDFYExpected.begin(),
                           [&](float f)
                           {
                               return f / totalSum;
                           });
            std::transform(hCDFYExpected.cbegin(), hCDFYExpected.cend(),
                           hCDFYExpected.begin(),
                           [&](float f)
                           {
                               return f / totalSum;
                           });
        }
        // Transform the PDF Back
        std::transform(hPDFYExpected.cbegin(), hPDFYExpected.cend(),
                       hPDFYExpected.begin(),
                       [&](float f)
                       {
                           return f * static_cast<float>(Y);
                       });
        // Checks
        for(uint32_t i = 0; i < PIX_COUNT; i++)
        {
            EXPECT_NEAR(hPDFXExpected[i], hPDFX[i], MathConstants::LargeEpsilon);
        }
        for(uint32_t i = 0; i < X_CDF_COUNT; i++)
        {
            EXPECT_NEAR(hCDFXExpected[i], hCDFX[i], MathConstants::LargeEpsilon);
        }
        for(uint32_t i = 0; i < Y; i++)
        {
            EXPECT_NEAR(hPDFYExpected[i], hPDFY[i], MathConstants::LargeEpsilon);
        }
        for(uint32_t i = 0; i < Y_CDF_COUNT; i++)
        {
            EXPECT_NEAR(hCDFYExpected[i], hCDFY[i], MathConstants::LargeEpsilon);
        }
    }
}
