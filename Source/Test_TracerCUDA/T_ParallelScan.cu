#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Vector.h"

#include "TracerCUDA/DeviceMemory.h"
#include "TracerCUDA/ParallelScan.cuh"

using ::testing::FloatEq;

TEST(ParallelScan, GenericExclusive)
{
    static constexpr int ElementCount = 50'000;

    DeviceMemory iDataIn(ElementCount * sizeof(int));
    DeviceMemory fDataIn(ElementCount * sizeof(float));
    DeviceMemory iDataOut(ElementCount * sizeof(int));
    DeviceMemory fDataOut(ElementCount * sizeof(float));
    std::fill_n(static_cast<int*>(iDataIn), ElementCount, 1);
    std::fill_n(static_cast<float*>(fDataIn), ElementCount, 1.0f);

    // Integer Kernel
    ExclusiveScanArrayGPU<int, ReduceAdd<int>>
    (
      static_cast<int*>(iDataOut),
      static_cast<int*>(iDataIn),
      ElementCount, 0
    );
    // Float Kernel
    ExclusiveScanArrayGPU<float, ReduceAdd<float>>
    (
      static_cast<float*>(fDataOut),
      static_cast<float*>(fDataIn),
      ElementCount, 0.0f
    );
    //EXPECT_TRUE(false);

    // Wait Kernel Finish and Check
    CUDA_CHECK(cudaDeviceSynchronize());

    for(int i = 0; i < ElementCount; i++)
    {
        EXPECT_EQ(static_cast<int*>(iDataOut)[i], i);
        EXPECT_FLOAT_EQ(static_cast<float*>(fDataOut)[i],
                        static_cast<float>(i));
    }
}

TEST(ParallelScan, GenericInclusive)
{
    static constexpr int ElementCount = 50'000;
    //static constexpr int ElementCount = 50;

    DeviceMemory iDataIn(ElementCount * sizeof(int));
    DeviceMemory fDataIn(ElementCount * sizeof(float));
    DeviceMemory iDataOut(ElementCount * sizeof(int));
    DeviceMemory fDataOut(ElementCount * sizeof(float));
    std::fill_n(static_cast<int*>(iDataIn), ElementCount, 1);
    std::fill_n(static_cast<float*>(fDataIn), ElementCount, 1.0f);

    // Integer Kernel
    InclusiveScanArrayGPU<int, ReduceAdd<int>>
    (
      static_cast<int*>(iDataOut),
      static_cast<int*>(iDataIn),
      ElementCount
    );
    // Float Kernel
    InclusiveScanArrayGPU<float, ReduceAdd<float>>
    (
      static_cast<float*>(fDataOut),
      static_cast<float*>(fDataIn),
      ElementCount
    );

    // Wait Kernel Finish and Check
    CUDA_CHECK(cudaDeviceSynchronize());

    for(int i = 0; i < ElementCount; i++)
    {
        EXPECT_EQ(static_cast<int*>(iDataOut)[i], i + 1);
        EXPECT_FLOAT_EQ(static_cast<float*>(fDataOut)[i],
                        static_cast<float>(i + 1));
    }
}