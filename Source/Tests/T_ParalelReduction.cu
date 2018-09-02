#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Vector.h"
#include "RayLib/DeviceMemory.h"
#include "RayLib/ParallelReduction.cuh"

using ::testing::ElementsAre;
using ::testing::FloatEq;

TEST(ParalelReduction, Construction)
{

	static constexpr int ElementCount = 5'000'000;

	DeviceMemory v2DataIn(ElementCount * sizeof(int));
	
	int result = 0;
	KCReduceArray<int, ReduceAdd, StaticThreadPerBlock1D, cudaMemcpyDeviceToHost>
	(
		result,
		static_cast<int*>(v2DataIn), 
		0, 
		ElementCount,
		0);

	EXPECT_EQ(ElementCount, result);
}