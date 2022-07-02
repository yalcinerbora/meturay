
#include "TracerCUDA/CudaSystem.h"
#include "TracerCUDA/CudaSystem.hpp"
#include "TracerCUDA/DeviceMemory.h"
#include "TracerCUDA/GPUReconFilterBox.h"

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

using Implementations = ::testing::Types<ReconFilterTestParams<GPUReconFilter<ReconBoxFilterFunctor>>;

TYPED_TEST_SUITE(ReconFilterTest, Implementations);

TYPED_TEST(ReconFilterTest, SmallSizeTest)
{
    CudaSystem system;
    ASSERT_EQ(CudaError::OK, system.Initialize());

    using GPUReconFilterT = typename TypeParam::ReconFilter;

    GPUReconFilterT
}