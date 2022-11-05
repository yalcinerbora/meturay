#include "TracerCUDA/DeviceMemory.h"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/ProductSampler.cuh"
#include "TracerCUDA/GPUMetaSurfaceGenerator.h"
#include "TracerCUDA/RNGIndependent.cuh"

#include "RayLib/Constants.h"

#include <gtest/gtest.h>
#include <random>
#include <numeric>

template <uint32_t TPB, uint32_t X, uint32_t Y>
__global__ __launch_bounds__(TPB)
void KCProductSamplerTest(RNGeneratorGPUI** gRNGs,
                          const GPUMetaSurfaceGeneratorGroup g)
{
    using ProductSamplerType = ProductSampler<TPB, X, Y, 8, 8>;
    __shared__ typename ProductSamplerType::SharedStorage sSharedMem;

    auto& rng = RNGAccessor::Acquire<RNGIndependentGPU>(gRNGs, LINEAR_GLOBAL_ID);


    ProductSamplerType productSampler(sSharedMem, nullptr, g);

    float pdf;
    Vector2f result = productSampler.SampleProduct(0, pdf, rng);
}

template <uint32_t TPB_VAL, uint32_t X_VAL, uint32_t Y_VAL>
struct ProductSamplerTestParams
{
    static constexpr uint32_t TPB               = TPB_VAL;
    static constexpr uint32_t X                 = X_VAL;
    static constexpr uint32_t Y                 = Y_VAL;
};

template <class T>
class ProductSamplerTest : public testing::Test
{};

//using Implementations = ::testing::Types<ProductSamplerTestParams<512, 64, 64>,
//                                         ProductSamplerTestParams<512, 32, 32>,
//                                         ProductSamplerTestParams<256, 16, 16>,
//                                         ProductSamplerTestParams<128, 8, 8>>;

using Implementations = ::testing::Types<ProductSamplerTestParams<512, 64, 64>>;

TYPED_TEST_SUITE(ProductSamplerTest, Implementations);

TYPED_TEST(ProductSamplerTest, Test)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    constexpr uint32_t TPB              = TypeParam::TPB;
    constexpr uint32_t X                = TypeParam::X;
    constexpr uint32_t Y                = TypeParam::Y;

    constexpr uint32_t SEED = 0;
    RNGIndependentCPU rngCPU(SEED, system);

    GPUMetaSurfaceGeneratorGroup g(nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   HitStructPtr());


    // PWC Initialization and Dump to Global Memory Call
    const CudaGPU& bestGPU = system.BestGPU();
    bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
                      //
                      KCProductSamplerTest<TPB, X, Y>,
                      //
                      rngCPU.GetGPUGenerators(bestGPU),
                      g);
}
