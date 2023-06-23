#include "TracerCUDA/DeviceMemory.h"
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/ProductSampler.cuh"
#include "TracerCUDA/GPUMetaSurfaceGenerator.h"
#include "TracerCUDA/RNGIndependent.cuh"

#include "RayLib/Constants.h"
#include "RayLib/CoordinateConversion.h"
#include "RayLib/HybridFunctions.h"

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
    auto ProjectionFunc = [](const Vector2i& localPixelId,
                             const Vector2i& segmentSize,
                             const Vector3f& surfNormal)
    {
        return Zero3f;
    };
    //auto WrapFunc = [](const Vector2i& sampleIndex,
    //                   const Vector2i& sampleCount)
    //{
    //    return Utility::CocentricOctohedralWrapInt(sampleIndex, sampleCount);
    //};

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
        Vector2f result = productSampler.SampleWithProduct(pdf, rng, -1,
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

//using Implementations = ::testing::Types<ProductSamplerTestParams<128, 2, 2>>;

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

TYPED_TEST(ProductSamplerTest, SamplePWCZeroVariance)
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
    for(int ik = 0; ik < KERNEL_CALL_COUNT; ik++)
    {
        // Populate the region (do uniform test for the first time)
        std::vector<float> hTexture(X * Y, 10.0f);
        if(ik != 0)
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

//TYPED_TEST(ProductSamplerTest, SamplePWLZeroVariance)
//{
//    CudaSystem system;
//    ASSERT_EQ(CudaError::OK, system.Initialize());
//    const CudaGPU& bestGPU = system.BestGPU();
//
//    static constexpr int32_t TPB = TypeParam::TPB;
//    static constexpr int32_t X = TypeParam::X;
//    static constexpr int32_t Y = TypeParam::Y;
//    static constexpr int32_t PX = TypeParam::PX;
//    static constexpr int32_t PY = TypeParam::PX;
//    static constexpr int32_t SAMPLE_PER_KERNEL = 8;
//    // Generate your RNG
//    constexpr uint32_t SEED = 0;
//    RNGIndependentCPU rngCPU(SEED, bestGPU);
//    // CPU RNG
//    std::mt19937 rng;
//    rng.seed(SEED);
//    std::uniform_real_distribution<float> uniformDist(0.0f, 5.0f);
//
//    // Create output sample / pdf array
//    // and input
//    DeviceMemory testMem;
//    float*      dInputTexture;
//    float*      dOutputPDFs;
//    Vector2f*   dOutputUVs;
//    GPUMemFuncs::AllocateMultiData(std::tie(dInputTexture, dOutputPDFs, dOutputUVs),
//                                   testMem,
//                                   {X * Y, SAMPLE_PER_KERNEL, SAMPLE_PER_KERNEL});
//
//    // Return memory
//    std::vector<float> hPDFResults(SAMPLE_PER_KERNEL);
//    std::vector<Vector2f> hUVResults(SAMPLE_PER_KERNEL);
//
//    static constexpr int32_t KERNEL_CALL_COUNT = 2;
//    for(int i = 1; i < KERNEL_CALL_COUNT; i++)
//    {
//        // Populate the region (do uniform test for the first time)
//        std::vector<float> hTexture(X * Y, 10.0f);
//        if(i != 0)
//        {
//            //std::for_each(hTexture.begin(), hTexture.end(),
//            //              [&uniformDist, &rng](float& f) { f = uniformDist(rng); });
//            hTexture = {1, 2, 3, 4};
//        }
//        // Set NaN to outputs just to be sure
//        CUDA_CHECK(cudaMemset(dOutputPDFs, 0xFF, sizeof(float) * SAMPLE_PER_KERNEL));
//        CUDA_CHECK(cudaMemset(dOutputUVs, 0xFF, sizeof(Vector2f) * SAMPLE_PER_KERNEL));
//
//        // Copy texture to GPU
//        CUDA_CHECK(cudaMemcpy(dInputTexture, hTexture.data(),
//                              sizeof(float) * X * Y,
//                              cudaMemcpyHostToDevice));
//
//        // Call the sample kernel
//        auto data = bestGPU.GetKernelAttributes(reinterpret_cast<const void*>(KCProductSamplerSampleTest<TPB, X, Y, PX, PY>));
//        bestGPU.ExactKC_X(0, (cudaStream_t)0, TPB, 1,
//                          //
//                          KCProductSamplerSampleTest<TPB, X, Y, PX, PY>,
//                          //
//                          dOutputPDFs,
//                          dOutputUVs,
//                          // I-O
//                          rngCPU.GetGPUGenerators(bestGPU),
//                          // Input
//                          dInputTexture,
//                          GPUMetaSurfaceGeneratorGroup(nullptr, nullptr, nullptr,
//                                                       nullptr, nullptr,
//                                                       HitStructPtr()),
//                          SAMPLE_PER_KERNEL,
//                          false);
//
//        // While it is doing its job (at least on release configuration)
//        // Calculate the integral result of the texture
//        // Integral here is not easy to find (not everything is trapezoid now)
//        // also we have one less trapezoid than the rectangles, we do wrap here however
//        // since product sampler represents spherical field.
//        auto TrapezoidArea = [](float a, float b, float h)
//        {
//            return (a + b) * 0.5f * h;
//        };
//        auto LinearizeXY = [](Vector2i i, int32_t width)
//        {
//           return i[1] * width + i[0];
//        };
//        // Calculate row by row pwl functions
//        std::vector<float> hRowAreas(Y + 1, 0.0f);
//        for(int j = 0; j <= Y; j++)
//        for(int i = 0; i < X; i++)
//        {
//            Vector2i cur = Utility::CocentricOctohedralWrapInt(Vector2i(i    , j), Vector2i(X, Y));
//            Vector2i next  = Utility::CocentricOctohedralWrapInt(Vector2i(i + 1, j), Vector2i(X, Y));
//
//            int32_t iLinear = LinearizeXY(cur, X);
//            int32_t iNextLinear = LinearizeXY(next, X);
//
//            hRowAreas[j] += TrapezoidArea(hTexture[iLinear], hTexture[iNextLinear], 1.0f / X);
//        }
//        float totalArea = 0.0f;
//        for(int j = 0; j < Y; j++)
//        {
//            totalArea += TrapezoidArea(hRowAreas[j], hRowAreas[j + 1], 1.0f / Y);
//        }
//        float expectedIntegral = totalArea;
//
//        // Copy and check the results
//        CUDA_CHECK(cudaMemcpy(hPDFResults.data(), dOutputPDFs,
//                              sizeof(float) * SAMPLE_PER_KERNEL,
//                              cudaMemcpyDeviceToHost));
//        CUDA_CHECK(cudaMemcpy(hUVResults.data(), dOutputUVs,
//                              sizeof(Vector2f) * SAMPLE_PER_KERNEL,
//                              cudaMemcpyDeviceToHost));
//
//
//        // TEST
//        std::vector<float> pdfX((X + 1) * (Y + 1), 0.0f);
//        for(int j = 0; j <= Y; j++)
//        for(int i = 0; i <= X; i++)
//        {
//            int32_t writeIndex = LinearizeXY(Vector2i(i, j), (X + 1));
//            Vector2i curWrapped = Utility::CocentricOctohedralWrapInt(Vector2i(i, j),
//                                                                      Vector2i(X, Y));
//            int32_t wLinear = LinearizeXY(curWrapped, X);
//
//            //METU_LOG("{} :{} = {} / {} * X", writeIndex,
//            //         hTexture[wLinear] / hRowAreas[j] * X,
//            //         hTexture[wLinear], hRowAreas[j]);
//            pdfX[writeIndex] = hTexture[wLinear] / hRowAreas[j];// *X;
//        }
//
//        std::vector<float> pdfY((Y + 1), 0.0f);
//        for(int j = 0; j <= Y; j++)
//        {
//            pdfY[j] = hRowAreas[j] / totalArea;// *Y;
//        }
//
//        // Test
//        Vector2f testIndex = Vector2f(0.2f, 0.2f);
//        Vector2f prev = Vector2f(std::trunc(testIndex[0]),
//                                 std::trunc(testIndex[1]));
//        Vector2f interp = testIndex - prev;
//        assert(interp >= Zero2f && interp < Vector2f(1.0f));
//
//        Vector2i prevI = Vector2i(prev);
//
//        Vector2i x00 = Utility::CocentricOctohedralWrapInt(prevI + Vector2i(0, 0), Vector2i(X, Y));
//        Vector2i x10 = Utility::CocentricOctohedralWrapInt(prevI + Vector2i(1, 0), Vector2i(X, Y));
//        Vector2i x01 = Utility::CocentricOctohedralWrapInt(prevI + Vector2i(0, 1), Vector2i(X, Y));
//        Vector2i x11 = Utility::CocentricOctohedralWrapInt(prevI + Vector2i(1, 1), Vector2i(X, Y));
//
//        // Function Value
//        float funcValX0 = HybridFuncs::Lerp(hTexture[LinearizeXY(x00, X)],
//                                            hTexture[LinearizeXY(x10, X)], interp[0]);
//        float funcValX1 = HybridFuncs::Lerp(hTexture[LinearizeXY(x01, X)],
//                                            hTexture[LinearizeXY(x11, X)], interp[0]);
//        float funcVal = HybridFuncs::Lerp(funcValX0, funcValX1, interp[1]);
//
//
//        //x00 = prevI + Vector2i(0, 0) % Vector2f((X + 2), (Y + 2));
//        //x10 = prevI + Vector2i(0, 0) % Vector2f((X + 2), (Y + 2));
//        //x01 = prevI + Vector2i(0, 0) % Vector2f((X + 2), (Y + 2));
//        //x11 = prevI + Vector2i(0, 0) % Vector2f((X + 2), (Y + 2));
//
//
//        // PDF Calculation
//        //float pdfXY_00 = pdfY[prevI[1]] * pdfX[LinearizeXY(x00, X + 1)];
//        //float pdfXY_10 = pdfY[prevI[1]] * pdfX[LinearizeXY(x10, X + 1)];
//        //float pdfXY_01 = pdfY[prevI[1] + 1] * pdfX[LinearizeXY(x01, X + 1)];
//        //float pdfXY_11 = pdfY[prevI[1] + 1] * pdfX[LinearizeXY(x11, X + 1)];
//        float pdfXY_00 = pdfX[LinearizeXY(x00, X + 1)];
//        float pdfXY_10 = pdfX[LinearizeXY(x10, X + 1)];
//        float pdfXY_01 = pdfX[LinearizeXY(x01, X + 1)];
//        float pdfXY_11 = pdfX[LinearizeXY(x11, X + 1)];
//        // Bilerp to find pdf
//        float pdf = HybridFuncs::Lerp(pdfY[prevI[1] + 0] * HybridFuncs::Lerp(pdfXY_00, pdfXY_10, interp[0]),
//                                      pdfY[prevI[1] + 1] * HybridFuncs::Lerp(pdfXY_01, pdfXY_11, interp[0]),
//                                      interp[1]);
//
//        float result = funcVal / pdf;
//        float test = hTexture[0] / (pdfY[0] * pdfX[0]);
//
//
//        float total = 0.0f;
//        for(int i = 0; i < SAMPLE_PER_KERNEL; i++)
//        {
//            Vector2i uvInt(hUVResults[i] * Vector2f(X, Y));
//            int32_t indexLinear = uvInt[1] * X + uvInt[0];
//
//            // Sample does two-layer PWC Distribution,
//            // Kernel does not do product on the outer layer
//            // Thus result should have zero variance
//            // meaning the division should give the exact value of the integral
//            float foundIntegral = hTexture[indexLinear] / hPDFResults[i];
//            EXPECT_NEAR(expectedIntegral, foundIntegral, MathConstants::LargeEpsilon);
//
//            // We might as well do Monte Carlo here
//            total += foundIntegral;
//        }
//        // Monte Carlo Result should be equal as well.
//        EXPECT_NEAR(expectedIntegral, total / SAMPLE_PER_KERNEL, MathConstants::VeryLargeEpsilon);
//    }
//}