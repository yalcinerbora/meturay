
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DeviceMemory.h"
#include "TracerCUDA/ImageMemory.h"

#include "TracerCUDA/GPUReconFilterBox.h"
//#include "TracerCUDA/GPUReconFilterTent.h"

#include "RayLib/Constants.h"

#include <gtest/gtest.h>
#include <random>
#include <numeric>

template <class Filter>
struct ReconFilterTestParams
{
    using ReconFilter = typename Filter;
};

template <class T>
class ReconFilterTest : public testing::Test
{};

using Implementations = ::testing::Types<ReconFilterTestParams<GPUReconFilterBox>>;

TYPED_TEST_SUITE(ReconFilterTest, Implementations);

TYPED_TEST(ReconFilterTest, Basic)
{
    using GPUReconFilterT = typename TypeParam::ReconFilter;
    // Statics
    static constexpr uint32_t SAMPLE_COUNT = 512;
    static constexpr Vector2i RESOLUTION = Vector2i{320, 180};
    static constexpr uint32_t PIXEL_COUNT = RESOLUTION.Multiply();
    static constexpr float RADIUS = 0.5f;
    const Options emptyOptions;
    // RNG
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<float> uniformDistX(0.0f, static_cast<float>(RESOLUTION[0]));
    std::uniform_real_distribution<float> uniformDistY(0.0f, static_cast<float>(RESOLUTION[1]));

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    GPUReconFilterT filter(RADIUS, emptyOptions);

    // Generate Some Samples
    std::vector<Vector4f> values(SAMPLE_COUNT, Vector4f(1.0f));
    std::vector<Vector2f> coordinates(SAMPLE_COUNT);

    std::for_each(coordinates.begin(), coordinates.end(),
                  [&](Vector2f& coord)
                  {
                      coord[0] = uniformDistX(rng);
                      coord[1] = uniformDistY(rng);
                  });


    // Device Stuff
    ImageMemory imgMem(Zero2i, RESOLUTION, RESOLUTION,
                       PixelFormat::RGBA_FLOAT);
    // Values and Samples
    Vector4f* dValues;
    Vector2f* dCoords;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dValues, dCoords),
                                   mem, {SAMPLE_COUNT, SAMPLE_COUNT});

    CUDA_CHECK(cudaMemcpy(dValues, values.data(),
                          sizeof(Vector4f) * SAMPLE_COUNT,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCoords, coordinates.data(),
                          sizeof(Vector2f) * SAMPLE_COUNT,
                          cudaMemcpyHostToDevice));


    filter.FilterToImg(imgMem, dValues, dCoords, SAMPLE_COUNT,
                       system);

    // Load Image Memory to CPU
    ImageGMem<Vector4f> imgMemPtrs = imgMem.GMem<Vector4f>();

    std::vector<Vector4f> pixels(PIXEL_COUNT);
    std::vector<float> samples(PIXEL_COUNT);

    CUDA_CHECK(cudaMemcpy(pixels.data(), imgMemPtrs.gPixels,
                          sizeof(Vector4f) * PIXEL_COUNT,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(samples.data(), imgMemPtrs.gSampleCounts,
                          sizeof(float) * PIXEL_COUNT,
                          cudaMemcpyDeviceToHost));


}