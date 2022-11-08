#include "TracerCUDA/DeviceMemory.h"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/ProductSampler.cuh"
#include "TracerCUDA/GPUMetaSurfaceGenerator.h"
#include "TracerCUDA/RNGIndependent.cuh"

#include "RayLib/Constants.h"
#include "RayLib/CoordinateConversion.h"

#include <gtest/gtest.h>
#include <random>
#include <numeric>

template <int32_t TPB, int32_t X, int32_t Y, int32_t PX, int32_t PY>
__global__ __launch_bounds__(TPB)
void KCProductSamplerInitTest(// Output
                              float* gRadianceField,
                              float* gSmallRadianceField,
                              float* gNormConstants,
                              // Inputs
                              const GPUMetaSurfaceGeneratorGroup g)
{
    using ProductSamplerType = ProductSampler<TPB, X, Y, PX, PY>;
    __shared__ typename ProductSamplerType::SharedStorage sSharedMem;

    float radiances[ProductSamplerType::RADIANCE_PER_THREAD];
    for(int i = 0; i < ProductSamplerType::RADIANCE_PER_THREAD; i++)
    {
        radiances[i] = static_cast<float>(i * TPB + threadIdx.x);
    }

    // Generate the product and check if the reduced image is OK
    ProductSamplerType productSampler(sSharedMem, radiances, nullptr, g);
    productSampler.DumpRadianceField(gRadianceField);
    productSampler.DumpSmallRadianceField(gSmallRadianceField);
    productSampler.DumpNormalizationConstants(gNormConstants);
}

template <int32_t TPB, int32_t X, int32_t Y, int32_t PX, int32_t PY>
__global__ __launch_bounds__(TPB)
void KCProductSamplerSampleTest(// Outputs
                                float* dOutputPDFs,
                                Vector2f* dOutputUVs,
                                // I-O
                                RNGeneratorGPUI** gRNGs,
                                // Inputs
                                const float* dInputTexture,
                                const GPUMetaSurfaceGeneratorGroup g,
                                int32_t samplePerThread)
{
    static constexpr int32_t WARP_PER_BLOCK = TPB / WARP_SIZE;
    static_assert((TPB % WARP_SIZE) == 0, "TPB should evenly be divisible by warpSize(32)");
    using ProductSamplerType = ProductSampler<TPB, X, Y, PX, PY>;
    // We will not do product sampling so give an invalid kernel
    auto ProjectionFunc = [&](const Vector2i& localPixelId,
                              const Vector2i& segmentSize)
    {
        return Zero3f;
    };
    // Product samplers shared memory
    __shared__ typename ProductSamplerType::SharedStorage sSharedMem;
    // Per-thread RNG
    auto& rng = RNGAccessor::Acquire<RNGIndependentGPU>(gRNGs, LINEAR_GLOBAL_ID);
    // Load the radiances
    float radiances[ProductSamplerType::RADIANCE_PER_THREAD];
    for(int i = 0; i < ProductSamplerType::RADIANCE_PER_THREAD; i++)
    {
        int32_t globalIndex = i * TPB + threadIdx.x;
        radiances[i] = dInputTexture[globalIndex];
    }
    // Construct the product sampler
    ProductSamplerType productSampler(sSharedMem, radiances, nullptr, g);
    // Warp-stride loop
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    bool isWarpLeader = (laneId == 0);
    for(int i = warpId; i < samplePerThread; i += WARP_PER_BLOCK)
    {
        float pdf;
        Vector2f result = productSampler.SampleProduct(pdf, rng, -1,
                                                       ProjectionFunc);
        if(isWarpLeader)
        {
            // We can directly use i here kernel is invoked for only one block
            dOutputPDFs[i] = pdf;
            dOutputUVs[i] = result;
        }
    }
}

template <int32_t TPB_VAL, int32_t X_VAL, int32_t Y_VAL>
struct ProductSamplerTestParams
{
    static constexpr int32_t TPB               = TPB_VAL;
    static constexpr int32_t X                 = X_VAL;
    static constexpr int32_t Y                 = Y_VAL;
    static constexpr int32_t PX                = 8;
    static constexpr int32_t PY                = 8;
};

template <class T>
class ProductSamplerTest : public testing::Test
{};

using Implementations = ::testing::Types<ProductSamplerTestParams<256, 64, 64>,
                                         ProductSamplerTestParams<256, 32, 32>,
                                         ProductSamplerTestParams<256, 16, 16>,
                                         ProductSamplerTestParams<128, 8, 8>>;

//using Implementations = ::testing::Types<ProductSamplerTestParams<256, 8, 8>>;

TYPED_TEST_SUITE(ProductSamplerTest, Implementations);

TYPED_TEST(ProductSamplerTest, Init)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    constexpr int32_t TPB              = TypeParam::TPB;
    constexpr int32_t X                = TypeParam::X;
    constexpr int32_t Y                = TypeParam::Y;
    constexpr int32_t PX               = TypeParam::PX;
    constexpr int32_t PY               = TypeParam::PY;

    float* dRadianceField;
    float* dSmallRadianceField;
    float* dNormalizationConstants;
    DeviceMemory radianceFieldMemory;
    GPUMemFuncs::AllocateMultiData(std::tie(dRadianceField,
                                            dSmallRadianceField,
                                            dNormalizationConstants),
                                   radianceFieldMemory,
                                   {X * Y, PX * PY, PX * PY});

    GPUMetaSurfaceGeneratorGroup g(nullptr, nullptr, nullptr,
                                   nullptr, nullptr,
                                   HitStructPtr());

    const CudaGPU& bestGPU = system.BestGPU();
    bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                      //
                      KCProductSamplerInitTest<TPB, X, Y, PX, PY>,
                      //
                      dRadianceField,
                      dSmallRadianceField,
                      dNormalizationConstants,
                      GPUMetaSurfaceGeneratorGroup(nullptr, nullptr, nullptr,
                                                   nullptr, nullptr,
                                                   HitStructPtr()));

    std::vector<float> hRadianceField(X * Y);
    std::vector<float> hSmallRadianceField(PX * PY);
    std::vector<float> hNormalizationConstants(PX * PY);
    CUDA_CHECK(cudaMemcpy(hRadianceField.data(), dRadianceField,
                          X * Y * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hSmallRadianceField.data(), dSmallRadianceField,
                          PX * PY * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hNormalizationConstants.data(), dNormalizationConstants,
                          PX * PY * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < hRadianceField.size(); i++)
    {
        EXPECT_FLOAT_EQ(hRadianceField[i], static_cast<float>(i));
    }
    for(int j = 0; j < PY; j++)
    for(int i = 0; i < PX; i++)
    {
        int outerIdLinear = j * PX + i;

        float maxVal = 0.0f;
        float total = 0.0f;
        static constexpr Vector2i innerCount = Vector2i(X / PX, Y / PY);
        for(int k = 0; k < innerCount[1]; k++)
        for(int l = 0; l < innerCount[0]; l++)
        {
            Vector2i id(i * innerCount[0] + l,
                        j * innerCount[1] + k);
            int idLinear = id[1] * X + id[0];
            total += hRadianceField[idLinear];
            maxVal = max(maxVal, hRadianceField[idLinear]);
        }
        total /= static_cast<float>(innerCount.Multiply());

        EXPECT_FLOAT_EQ(hSmallRadianceField[outerIdLinear], total);
        EXPECT_FLOAT_EQ(hNormalizationConstants[outerIdLinear], maxVal);
    }
}

TYPED_TEST(ProductSamplerTest, SampleZeroVariance)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    const CudaGPU& bestGPU = system.BestGPU();

    static constexpr int32_t TPB = TypeParam::TPB;
    static constexpr int32_t X = TypeParam::X;
    static constexpr int32_t Y = TypeParam::Y;
    static constexpr int32_t PX = TypeParam::PX;
    static constexpr int32_t PY = TypeParam::PX;
    static constexpr int32_t SAMPLE_PER_KERNEL = 1024;
    // Generate your RNG
    constexpr uint32_t SEED = 0;
    RNGIndependentCPU rngCPU(SEED, bestGPU);
    // CPU RNG
    std::mt19937 rng;
    rng.seed(SEED);
    std::uniform_real_distribution<float> uniformDist(0.0f, 5.0f);

    // Create output sample / pdf array
    // and input
    DeviceMemory testMem;
    float*      dInputTexture;
    float*      dOutputPDFs;
    Vector2f*   dOutputUVs;
    GPUMemFuncs::AllocateMultiData(std::tie(dInputTexture, dOutputPDFs, dOutputUVs),
                                   testMem,
                                   {X * Y, SAMPLE_PER_KERNEL, SAMPLE_PER_KERNEL});

    // Return memory
    std::vector<float> hPDFResults(SAMPLE_PER_KERNEL);
    std::vector<Vector2f> hUVResults(SAMPLE_PER_KERNEL);

    static constexpr int32_t KERNEL_CALL_COUNT = 128;
    for(int i = 0; i < KERNEL_CALL_COUNT; i++)
    {
        // Populate the region (do uniform test for the first time)
        std::vector<float> hTexture(X * Y, 10.0f);
        if(i != 0)
        {
            std::for_each(hTexture.begin(), hTexture.end(),
                          [&uniformDist, &rng](float& f) { f = uniformDist(rng); });
        }
        // Set NaN to outputs just to be sure
        CUDA_CHECK(cudaMemset(dOutputPDFs, 0xFF, sizeof(float) * SAMPLE_PER_KERNEL));
        CUDA_CHECK(cudaMemset(dOutputUVs, 0xFF, sizeof(Vector2f) * SAMPLE_PER_KERNEL));

        // Copy texture to GPU
        CUDA_CHECK(cudaMemcpy(dInputTexture, hTexture.data(),
                              sizeof(float) * X * Y,
                              cudaMemcpyHostToDevice));

        // Call the sample kernel
        auto data = bestGPU.GetKernelAttributes(reinterpret_cast<const void*>(KCProductSamplerSampleTest<TPB, X, Y, PX, PY>));
        bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                          //
                          KCProductSamplerSampleTest<TPB, X, Y, PX, PY>,
                          //
                          dOutputPDFs,
                          dOutputUVs,
                          // I-O
                          rngCPU.GetGPUGenerators(bestGPU),
                          // Input
                          dInputTexture,
                          GPUMetaSurfaceGeneratorGroup(nullptr, nullptr, nullptr,
                                                       nullptr, nullptr,
                                                       HitStructPtr()),
                          SAMPLE_PER_KERNEL);


        // While it is doing its job (at least on release configuration)
        // Calculate the integral result of the texture
        float expectedIntegral = std::reduce(hTexture.cbegin(), hTexture.cend(), 0.0f);
        expectedIntegral /= X;
        expectedIntegral /= Y;
        // Copy and check the results
        CUDA_CHECK(cudaMemcpy(hPDFResults.data(), dOutputPDFs,
                              sizeof(float) * SAMPLE_PER_KERNEL,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hUVResults.data(), dOutputUVs,
                              sizeof(Vector2f) * SAMPLE_PER_KERNEL,
                              cudaMemcpyDeviceToHost));


        float total = 0.0f;
        for(int i = 0; i < SAMPLE_PER_KERNEL; i++)
        {
            Vector2i uvInt(hUVResults[i] * Vector2f(X, Y));
            int32_t indexLinear = uvInt[1] * X + uvInt[0];

            // Sample does two-layer PWC Distribution,
            // Kernel does not do product on the outer layer
            // Thus result should have zero variance
            // meaning the division should give the exact value of the integral
            float foundIntegral = hTexture[indexLinear] / hPDFResults[i];
            EXPECT_NEAR(expectedIntegral, foundIntegral, MathConstants::LargeEpsilon);

            // We might as well do Monte Carlo here
            total += foundIntegral;
        }
        // Monte Carlo Result should be equal as well.
        EXPECT_NEAR(expectedIntegral, total / SAMPLE_PER_KERNEL, MathConstants::VeryLargeEpsilon);
    }
}