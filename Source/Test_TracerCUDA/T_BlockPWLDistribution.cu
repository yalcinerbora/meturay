#include "TracerCUDA/GPUBlockPWLDistribution.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DeviceMemory.h"

#include "RayLib/Constants.h"

#include <gtest/gtest.h>
#include <random>
#include <numeric>

template <uint32_t TPB, uint32_t X, uint32_t Y>
__global__ __launch_bounds__(TPB)
void KCPiecewiseLinearDistInitCheck(float* gPDFXOut,
                                    float* gCDFXOut,
                                    float* gPDFYOut,
                                    float* gCDFYOut,
                                    const float* gData)
{
    using BlockPWL2D = BlockPWLDistribution2D<TPB, X, Y>;
    static constexpr auto DATA_PER_THREAD = BlockPWL2D::DATA_PER_THREAD;
    static constexpr auto PIXEL_COUNT = BlockPWL2D::PIX_COUNT;

    // Allocate shared memory for Block Operations
    __shared__ typename BlockPWL2D::TempStorage sPWLMem;

    const uint32_t threadId = threadIdx.x;

    float data[DATA_PER_THREAD];
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        data[i] = (threadId < PIXEL_COUNT) ? gData[i * TPB + threadId] : 0.0f;
    }

    // Init the class
    BlockPWL2D dist2D(sPWLMem, data);
    // Directly dump the calculated PDF / CDF
    dist2D.DumpSharedMem(gPDFXOut,
                         gCDFXOut,
                         gPDFYOut,
                         gCDFYOut);

}

template <uint32_t TPB_VAL, uint32_t X_VAL, uint32_t Y_VAL>
struct BlockPWL2DTestParams
{
    static constexpr uint32_t TPB = TPB_VAL;
    static constexpr uint32_t X = X_VAL;
    static constexpr uint32_t Y = Y_VAL;
    static constexpr uint32_t PIX_COUNT = X * Y;
    static constexpr uint32_t DATA_PER_BLOCK = PIX_COUNT / TPB;
    static constexpr uint32_t X_CDF_COUNT = X * Y;
    static constexpr uint32_t Y_CDF_COUNT = Y;

};

template <class T>
class BlockPWL2DTest : public testing::Test
{};

using Implementations = ::testing::Types<BlockPWL2DTestParams<512, 64, 64>,
                                         BlockPWL2DTestParams<512, 64, 32>,
                                         BlockPWL2DTestParams<512, 32, 32>,
                                         BlockPWL2DTestParams<256, 32, 16>,
                                         BlockPWL2DTestParams<256, 16, 16>,
                                         BlockPWL2DTestParams<128, 16, 8>>;

TYPED_TEST_SUITE(BlockPWL2DTest, Implementations);

TYPED_TEST(BlockPWL2DTest, BasicInit)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    constexpr uint32_t TPB = TypeParam::TPB;
    constexpr uint32_t X = TypeParam::X;
    constexpr uint32_t Y = TypeParam::Y;
    constexpr uint32_t PIX_COUNT = TypeParam::PIX_COUNT;
    constexpr uint32_t X_CDF_COUNT = TypeParam::X_CDF_COUNT;
    constexpr uint32_t Y_CDF_COUNT = TypeParam::Y_CDF_COUNT;

    // Copy all ones to GPU
    std::vector<float> data(PIX_COUNT, 10.0f);
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

    // PWL Initialization and Dump to Global Memory Call
    const CudaGPU& bestGPU = system.BestGPU();
    bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                      //
                      KCPiecewiseLinearDistInitCheck<TPB, X, Y>,
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
    for(uint32_t i = 0; i < X; i++)
    {
        static constexpr float DELTA_X = 1.0f / (static_cast<float>(X) - 1.0f);

        // Technically we should check accumulated trapezoid area
        // but in this basic case it is a rectangle (entire portion)
        uint32_t index = j * X + i;
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
        // Technically we should check accumulated trapezoid area
        // but in this basic case it is a rectangle (entire portion)
        static constexpr float DELTA_Y = 1.0f / (static_cast<float>(Y) - 1.0f);
        float value = static_cast<float>(i) * DELTA_Y;
        EXPECT_FLOAT_EQ(value, hCDFY[i]);
    }
}

TYPED_TEST(BlockPWL2DTest, Stress)
{
    static constexpr uint32_t ITERATION_COUNT = 100;
    constexpr uint32_t TPB = TypeParam::TPB;
    constexpr uint32_t X = TypeParam::X;
    constexpr uint32_t Y = TypeParam::Y;
    constexpr uint32_t PIX_COUNT = TypeParam::PIX_COUNT;
    constexpr uint32_t X_CDF_COUNT = TypeParam::X_CDF_COUNT;
    constexpr uint32_t Y_CDF_COUNT = TypeParam::Y_CDF_COUNT;

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    auto TrapezoidArea = [](float a, float b, float h)
    {
        return (a + b) * h * 0.5f;
    };

    // CPU Allocations
    // Random PDFtion
    std::vector<float> hData(PIX_COUNT);
    // Results returned by the PWL
    std::vector<float> hPDFX(PIX_COUNT);
    std::vector<float> hCDFX(X_CDF_COUNT);
    std::vector<float> hPDFY(Y);
    std::vector<float> hCDFY(Y_CDF_COUNT);
    // Calculated Values
    std::vector<float> hPDFYExpected(Y);
    std::vector<float> hPDFXExpected(PIX_COUNT);
    std::vector<float> hCDFXExpected(X_CDF_COUNT);
    std::vector<float> hCDFYExpected(Y_CDF_COUNT);
    // Temp
    std::vector<float> hScratchpad;
    hScratchpad.reserve(std::max(X, Y));

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
                          KCPiecewiseLinearDistInitCheck<TPB, X, Y>,
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
        hScratchpad.resize(X);
        for(uint32_t j = 0; j < Y; j++)
        {
            static constexpr float DELTA_X = 1.0f / static_cast<float>(X - 1);
            const uint32_t dataRowStart = j * X;
            const uint32_t dataRowEnd = (j + 1) * X;

            hScratchpad[0] = 0.0f;
            for(uint32_t i = 1; i < X; i++)
            {
                hScratchpad[i] = TrapezoidArea(hData[dataRowStart + i - 1],
                                               hData[dataRowStart + i],
                                               DELTA_X);
            }
            std::copy(hData.cbegin() + dataRowStart,
                      hData.cbegin() + dataRowEnd,
                      hPDFXExpected.begin() + dataRowStart);

            uint32_t cdfRowStart = j * X;
            uint32_t cdfRowEnd = (j + 1) * X;
            std::inclusive_scan(hScratchpad.cbegin(), hScratchpad.cbegin() + X,
                                hCDFXExpected.begin() + cdfRowStart);
            float totalSum = hCDFXExpected[cdfRowEnd - 1];
            hPDFYExpected[j] = totalSum;

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
        }

        static constexpr float DELTA_Y = 1.0f / static_cast<float>(Y - 1);
        hScratchpad.resize(Y);
        hScratchpad[0] = 0.0f;
        for(uint32_t i = 1; i < Y; i++)
        {
            hScratchpad[i] = TrapezoidArea(hPDFYExpected[i - 1], hPDFYExpected[i], DELTA_Y);
        }
        std::inclusive_scan(hScratchpad.cbegin(), hScratchpad.cbegin() + Y,
                            hCDFYExpected.begin());
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