#include "TracerCUDA/GPUBlockPWCDistribution.cuh"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DeviceMemory.h"

#include "RayLib/Constants.h"

#include <gtest/gtest.h>
#include <random>
#include <numeric>

template <uint32_t TPB, uint32_t X, uint32_t Y>
__global__ __launch_bounds__(TPB)
void KCPiecewiseConstDistInitCheck(float* gPDFXOut,
                                   float* gCDFXOut,
                                   float* gPDFYOut,
                                   float* gCDFYOut,
                                   const float* gData)
{
    using BlockPWC2D = BlockPWCDistribution2D<TPB, X, Y>;
    static constexpr auto DATA_PER_THREAD = BlockPWC2D::DATA_PER_THREAD;
    static constexpr auto PIXEL_COUNT = BlockPWC2D::PIX_COUNT;

    // Allocate shared memory for Block Operations
    __shared__ typename BlockPWC2D::TempStorage sPWCMem;

    const uint32_t threadId = threadIdx.x;

    float data[DATA_PER_THREAD];
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        data[i] = (threadId < PIXEL_COUNT) ?  gData[i * TPB + threadId] : 0.0f;
    }

    // Init the class
    BlockPWC2D dist2D(sPWCMem, data);
    // Directly dump the calculated PDF / CDF
    dist2D.DumpSharedMem(gPDFXOut,
                         gCDFXOut,
                         gPDFYOut,
                         gCDFYOut);

}

template <uint32_t TPB_VAL, uint32_t X_VAL, uint32_t Y_VAL>
struct BlockPWC2DTestParams
{
    static constexpr uint32_t TPB               = TPB_VAL;
    static constexpr uint32_t X                 = X_VAL;
    static constexpr uint32_t Y                 = Y_VAL;
    static constexpr uint32_t PIX_COUNT         = X * Y;
    static constexpr uint32_t DATA_PER_BLOCK    = PIX_COUNT / TPB;
    static constexpr uint32_t X_CDF_COUNT       = (X + 1) * Y;
    static constexpr uint32_t Y_CDF_COUNT       = (Y + 1);

};

template <class T>
class BlockPWC2DTest : public testing::Test
{};

using Implementations = ::testing::Types<BlockPWC2DTestParams<512, 64, 64>,
                                         BlockPWC2DTestParams<512, 64, 32>,
                                         BlockPWC2DTestParams<512, 32, 32>,
                                         BlockPWC2DTestParams<256, 32, 16>,
                                         BlockPWC2DTestParams<256, 16, 16>,
                                         BlockPWC2DTestParams<128, 16, 8>>;

//using Implementations = ::testing::Types<BlockPWC2DTestParams<32, 2, 4>>;


TYPED_TEST_SUITE(BlockPWC2DTest, Implementations);

TYPED_TEST(BlockPWC2DTest, BasicInit)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    constexpr uint32_t TPB              = TypeParam::TPB;
    constexpr uint32_t X                = TypeParam::X;
    constexpr uint32_t Y                = TypeParam::Y;
    constexpr uint32_t PIX_COUNT        = TypeParam::PIX_COUNT;
    constexpr uint32_t X_CDF_COUNT      = TypeParam::X_CDF_COUNT;
    constexpr uint32_t Y_CDF_COUNT      = TypeParam::Y_CDF_COUNT;

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
                      KCPiecewiseConstDistInitCheck<TPB, X, Y>,
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

TYPED_TEST(BlockPWC2DTest, Stress)
{
    static constexpr uint32_t ITERATION_COUNT = 100;
    constexpr uint32_t TPB              = TypeParam::TPB;
    constexpr uint32_t X                = TypeParam::X;
    constexpr uint32_t Y                = TypeParam::Y;
    constexpr uint32_t PIX_COUNT        = TypeParam::PIX_COUNT;
    constexpr uint32_t X_CDF_COUNT      = TypeParam::X_CDF_COUNT;
    constexpr uint32_t Y_CDF_COUNT      = TypeParam::Y_CDF_COUNT;

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

    for(uint32_t ik = 0; ik < ITERATION_COUNT; ik++)
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
                          KCPiecewiseConstDistInitCheck<TPB, X, Y>,
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
