
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DeviceMemory.h"
#include "TracerCUDA/RaceSketch.cuh"

#include "RayLib/Constants.h"
#include "RayLib/ColorConversion.h"

#include <gtest/gtest.h>
#include <random>
#include <numeric>

#include "ImageIO/EntryPoint.h"
#include "ImageIO/ImageIOI.h"

__global__ __launch_bounds__(StaticThreadPerBlock1D)
void KCLoadImgAsSketch(RaceSketchGPU sketchGPU,
                       const Vector3f* gPixels,
                       const Vector2ui dim)
{
    static constexpr Vector3f ZERO = Zero3f;

    // Push UV Coords
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < dim.Multiply(); globalId += blockDim.x * gridDim.x)
    {
        uint32_t x = globalId % dim[0];
        uint32_t y = globalId / dim[0];

        const Vector2f dimF = Vector2f(dim[0], dim[1]);
        Vector2f uv = Vector2f((static_cast<float>(x) + 0.5f),
                               (static_cast<float>(y) + 0.5f));

        //uv /= dimF;
        //uv *= Vector2f(2.0f);
        //uv -= Vector2f(1.0f);
        //uv -= (dimF * 0.5f);

        //Vector3f pix = gPixels[globalId];
        //float lum = pix.Sum() * 0.33f;
        float lum = Utility::RGBToLuminance(gPixels[globalId]);

        if(lum <= 100)
            sketchGPU.AtomicAddData(ZERO, uv, lum);
    }
}

__global__ __launch_bounds__(StaticThreadPerBlock1D)
void KCReadSketch(float* gPixels,
                  const RaceSketchGPU sketchGPU,
                  const Vector2ui dim)
{
    static constexpr Vector3f ZERO = Zero3f;

    // Push UV Coords
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < dim.Multiply(); globalId += blockDim.x * gridDim.x)
    {
        uint32_t x = globalId % dim[0];
        uint32_t y = globalId / dim[0];

        const Vector2f dimF = Vector2f(dim[0], dim[1]);
        Vector2f uv = Vector2f((static_cast<float>(x) + 0.5f),
                               (static_cast<float>(y) + 0.5f));
        //uv /= dimF;
        //uv *= Vector2f(2.0f);
        //uv -= Vector2f(1.0f);
        //uv -= (dimF * 0.5f);

        float prob = sketchGPU.Probability(ZERO, uv);
        gPixels[globalId] = prob;
    }
}

TEST(RaceSketch2D, ImageApprox)
{
    static constexpr uint32_t HASH_COUNT = 4096;
    static constexpr uint32_t BIN_COUNT = 64;

    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    const ImageIOI* imgIO = ImageIOInstance();

    Vector2ui dim;
    PixelFormat fmt;
    std::vector<Byte> inPixels;
    ImageIOError e = imgIO->ReadImage(inPixels, fmt, dim, "test_sketch.exr");
    ASSERT_EQ(e, ImageIOError::OK);
    ASSERT_EQ(fmt, PixelFormat::RGB_FLOAT);

    size_t pixelCount = dim.Multiply();

    // Allocate GPU Mem
    float* dProbOut;
    Vector3f* dPixels;
    DeviceMemory mem;
    GPUMemFuncs::AllocateMultiData(std::tie(dProbOut, dPixels), mem,
                                   {pixelCount, pixelCount});
    CUDA_CHECK(cudaMemcpy(dPixels, inPixels.data(), sizeof(Vector3f) * pixelCount,
                          cudaMemcpyHostToDevice));

    RaceSketchCPU sketch(HASH_COUNT, BIN_COUNT, 15.5, 0);
    sketch.SetSceneExtent(AABB3f(Zero3f, Vector3f(1.0f)));

    const CudaGPU& gpu = system.BestGPU();

    // Convert to Sketch
    for(int i = 0; i < 1; i++)
    {

        gpu.GridStrideKC_X(0, 0, pixelCount,
                           //
                           KCLoadImgAsSketch,
                           //
                           sketch.SketchGPU(),
                           dPixels,
                           dim);
    }
    uint64_t total;
    std::vector<float> sketchData;
    sketch.GetSketchToCPU(sketchData, total);
    //
    //METU_LOG("Total {}", total);
    uint32_t zeroCount = 0;
    for(float f : sketchData)
        if(f == 0.0f)
            zeroCount++;
    METU_LOG("Zero Ratio %{:f}", static_cast<float>(zeroCount) / static_cast<float>(sketchData.size()));


    //for(int i = 0; i < HASH_COUNT; i++)
    //{
    //    METU_LOG("[{}] - [{}]",
    //             sketchData[i * 2 + 0],
    //             sketchData[i * 2 + 1]);
    //}

    // Convert Back
    gpu.GridStrideKC_X(0, 0, pixelCount,
                       //
                       KCReadSketch,
                       //
                       dProbOut,
                       sketch.SketchGPU(),
                       dim);


    std::vector<float> hSketchOut(pixelCount);
    CUDA_CHECK(cudaMemcpy(hSketchOut.data(), dProbOut, sizeof(float) * pixelCount,
                          cudaMemcpyDeviceToHost));

    // Write as Image
    e = imgIO->WriteImage(hSketchOut, dim, PixelFormat::R_FLOAT, ImageType::EXR, "testOUT_11.exr");
    ASSERT_EQ(e, ImageIOError::OK);
}