
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
    static constexpr uint32_t SAMPLE_COUNT = 1'000'000;
    static constexpr Vector2i RESOLUTION = Vector2i{1920, 1080};
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
    imgMem.Reportion(Zero2i, RESOLUTION, system);
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


    // Construct Expected Values Manually (using CPU)
    std::vector<Vector4f> expectedPixels(PIXEL_COUNT);
    std::vector<float> expectedSamples(PIXEL_COUNT);
    // Pixel Range
    const int32_t wh = static_cast<int32_t>(FilterRadiusToPixelWH(RADIUS));
    const typename GPUReconFilterT::FilterFunctor filterFunc(RADIUS);
    // Loop over every sample and add filtered values to images
    for(uint32_t i = 0; i < values.size(); i++)
    {
        Vector4f value = values[i];
        Vector2f sampleCoords = coordinates[i];

        // Sample Pixel Id etc.
        Vector2f samplePixId;
        Vector2f relImgCoords = Vector2f(modf(sampleCoords[0],
                                              &(samplePixId[0])),
                                         modf(sampleCoords[1],
                                              &(samplePixId[1])));
        Vector2i samplePixIdInt = Vector2i(samplePixId);

        // Wastefully do [-w,+w] range
        for(int y = -wh; y <= wh; y++)
        for(int x = -wh; x <= wh; x++)
        {
            Vector2f pixCoord = samplePixId + Vector2f(static_cast<float>(x),
                                                       static_cast<float>(y));
            Vector2f pixCenter = pixCoord + Vector2f(0.5f);

            Vector distVec = (sampleCoords - pixCenter);
            // If filter is in range of the pixel
            if(distVec.LengthSqr() <= RADIUS * RADIUS)
            {
                // Integer pixel id
                Vector2i pixId = Vector2i(samplePixIdInt[0] + x,
                                          samplePixIdInt[1] + y);

                bool pixXInside = (pixId[0] >= 0 && pixId[0] < RESOLUTION[0]);
                bool pixYInside = (pixId[1] >= 0 && pixId[1] < RESOLUTION[1]);
                // Out of bounds check
                if(pixXInside && pixYInside)
                {
                    // Linear Pixel Id
                    uint32_t pixelId = (pixId[1] * RESOLUTION[0] + pixId[0]);

                    float filterWeight = filterFunc(pixCenter, sampleCoords);
                    Vector4f weightedVal = filterWeight * value;

                    expectedPixels[pixelId] += weightedVal;
                    expectedSamples[pixelId] += filterWeight;
                }
            }
        }
    }

    // Now Compare
    for(uint32_t i = 0; i < pixels.size(); i++)
    {
        EXPECT_FLOAT_EQ(expectedPixels[i][0], pixels[i][0]);
        EXPECT_FLOAT_EQ(expectedPixels[i][1], pixels[i][1]);
        EXPECT_FLOAT_EQ(expectedPixels[i][2], pixels[i][2]);
        EXPECT_FLOAT_EQ(expectedPixels[i][3], pixels[i][3]);

        EXPECT_FLOAT_EQ(expectedSamples[i], samples[i]);
    }
}