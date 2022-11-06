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
void KCProductSamplerSampleTest(// I-O
                                RNGeneratorGPUI** gRNGs,
                                // Inputs
                                const GPUMetaSurfaceGeneratorGroup g)
{
    using ProductSamplerType = ProductSampler<TPB, X, Y, PX, PY>;
    __shared__ typename ProductSamplerType::SharedStorage sSharedMem;

    auto ProjectionFunc = [&](const Vector2i& localPixelId,
                              const Vector2i& segmentSize)
    {
        Vector2f st = Vector2f(localPixelId) + 0.5f;
        st /= Vector2f(segmentSize);
        Vector3 result = Utility::CocentricOctohedralToDirection(st);
        Vector3 dirYUp = Vector3(result[1], result[2], result[0]);
        return dirYUp;
    };

    auto& rng = RNGAccessor::Acquire<RNGIndependentGPU>(gRNGs, LINEAR_GLOBAL_ID);



    float radiances[ProductSamplerType::RADIANCE_PER_THREAD];
    for(int i = 0; i < ProductSamplerType::RADIANCE_PER_THREAD; i++)
    {
        radiances[i] = static_cast<float>(i * TPB + threadIdx.x);
    }

    ProductSamplerType productSampler(sSharedMem, radiances, nullptr, g);
    float pdf;
    Vector2f result = productSampler.SampleProduct(pdf, rng, 0,
                                                   ProjectionFunc);
}

template <int32_t TPB_VAL, int32_t X_VAL, int32_t Y_VAL>
struct ProductSamplerTestParams
{
    static constexpr int32_t TPB               = TPB_VAL;
    static constexpr int32_t X                 = X_VAL;
    static constexpr int32_t Y                 = Y_VAL;
};

template <class T>
class ProductSamplerTest : public testing::Test
{};

using Implementations = ::testing::Types<ProductSamplerTestParams<256, 64, 64>,
                                         ProductSamplerTestParams<256, 32, 32>,
                                         ProductSamplerTestParams<256, 16, 16>,
                                         ProductSamplerTestParams<128, 8, 8>>;

//using Implementations = ::testing::Types<ProductSamplerTestParams<32, 16, 16>>;

TYPED_TEST_SUITE(ProductSamplerTest, Implementations);

TYPED_TEST(ProductSamplerTest, Init)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    constexpr int32_t TPB              = TypeParam::TPB;
    constexpr int32_t X                = TypeParam::X;
    constexpr int32_t Y                = TypeParam::Y;
    constexpr int32_t PX               = 8;
    constexpr int32_t PY               = 8;

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
            Vector2i id(i + l, j + k);
            int idLinear = id[1] * X + id[0];
            total += hRadianceField[idLinear];
            maxVal = max(maxVal, hRadianceField[idLinear]);
        }
        total /= static_cast<float>(innerCount.Multiply());

        EXPECT_FLOAT_EQ(hSmallRadianceField[outerIdLinear], total);
        EXPECT_FLOAT_EQ(hNormalizationConstants[outerIdLinear], maxVal);
    }
}

TYPED_TEST(ProductSamplerTest, Sample)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());
    const CudaGPU& bestGPU = system.BestGPU();

    constexpr int32_t TPB = TypeParam::TPB;
    constexpr int32_t X = TypeParam::X;
    constexpr int32_t Y = TypeParam::Y;
    constexpr int32_t PX = 8;
    constexpr int32_t PY = 8;

    constexpr uint32_t SEED = 0;
    RNGIndependentCPU rngCPU(SEED, system);

    GPUMetaSurfaceGeneratorGroup g(nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   HitStructPtr());

    auto data = bestGPU.GetKernelAttributes(reinterpret_cast<const void*>(KCProductSamplerSampleTest<TPB, X, Y, PX, PY>));


    bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                      //
                      KCProductSamplerSampleTest<TPB, X, Y, PX, PY>,
                      //
                      rngCPU.GetGPUGenerators(bestGPU),
                      GPUMetaSurfaceGeneratorGroup(nullptr, nullptr, nullptr,
                                                   nullptr, nullptr,
                                                   HitStructPtr()));
}