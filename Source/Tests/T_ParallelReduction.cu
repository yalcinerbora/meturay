#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Vector.h"

#include "TracerLib/DeviceMemory.h"
#include "TracerLib/ParallelReduction.cuh"

using ::testing::FloatEq;

inline __device__ float CustomReduce(const float& a, const float& b)
{
	return 1.0f;
};

TEST(ParallelReduction, Generic)
{
	static constexpr int ElementCount = 5'000'000;

	DeviceMemory iDataIn(ElementCount * sizeof(int));
	DeviceMemory fDataIn(ElementCount * sizeof(float));
	std::fill_n(static_cast<int*>(iDataIn), ElementCount, 1);
	std::fill_n(static_cast<float*>(fDataIn), ElementCount, 1.0f);

	// Integer Kernel
	int resultInt = 0;
	KCReduceArray<int, ReduceAdd, cudaMemcpyDeviceToHost>
	(
		resultInt,
		static_cast<int*>(iDataIn),
		ElementCount,
		0
	);
	// Float Kernel
	float resultFloat = 0;
	KCReduceArray<float, ReduceAdd, cudaMemcpyDeviceToHost>
	(
		resultFloat,
		static_cast<float*>(fDataIn),
		ElementCount,
		0.0f
	);

	// Wait Kernel Finish and Check
	CUDA_CHECK(cudaDeviceSynchronize());
	EXPECT_EQ(ElementCount, resultInt);
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultFloat);
}

TEST(ParallelReduction, CustomFunction)
{
	static constexpr int ElementCount = 5'000'000;

	DeviceMemory fDataIn(ElementCount * sizeof(float));
	std::fill_n(static_cast<float*>(fDataIn), ElementCount, 1.0f);

	// Float Kernel
	float resultFloat = 0;
	KCReduceArray<float, CustomReduce, cudaMemcpyDeviceToHost>
	(
		resultFloat,
		static_cast<float*>(fDataIn),
		ElementCount,
		0.0f
	);

	// Wait Kernel Finish and Check
	CUDA_CHECK(cudaDeviceSynchronize());
	EXPECT_FLOAT_EQ(1.0f, resultFloat);
}

TEST(ParallelReduction, LargeStruct)
{
	static constexpr int ElementCount = 1'000'000;

	DeviceMemory m4x4DataIn(ElementCount * sizeof(Matrix4x4f));
	std::fill_n(static_cast<Matrix4x4f*>(m4x4DataIn), ElementCount, 1.0f);

	// Float Kernel
	Matrix4x4f resultMatrix = Matrix4x4f(0.0f);
	KCReduceArray<Matrix4x4f, ReduceAdd, cudaMemcpyDeviceToHost>
	(
		resultMatrix,
		static_cast<Matrix4x4f*>(m4x4DataIn),
		ElementCount,
		Zero4x4f
	);

	// Wait Kernel Finish and Check
	CUDA_CHECK(cudaDeviceSynchronize());

	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(0, 0));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(0, 1));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(0, 2));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(0, 3));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(1, 0));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(1, 1));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(1, 2));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(1, 3));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(2, 0));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(2, 1));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(2, 2));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(2, 3));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(3, 0));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(3, 1));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(3, 2));
	EXPECT_FLOAT_EQ(static_cast<float>(ElementCount), resultMatrix(3, 3));
}