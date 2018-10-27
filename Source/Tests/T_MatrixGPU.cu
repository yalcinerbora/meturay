#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Matrix.h"

#include "TracerLib/DeviceMemory.h"

#include <cuda_runtime.h>

using ::testing::ElementsAre;
using ::testing::FloatEq;

__global__ void KGlobalLoadStore(const Matrix2x2* m2Input,
								 const Matrix3x3* m3Input,
								 const Matrix4x4* m4Input,
								 const float* fInput,

								 Matrix2x2* m2Output,
								 Matrix3x3* m3Output,
								 Matrix4x4* m4Output,
								 float* fM2Output,
								 float* fM3Output,
								 float* fM4Output)
{
	unsigned int gId = threadIdx.x + blockIdx.x * blockDim.x;

	// Testing Direct Load and Store
	Matrix2x2 m0 = m2Input[gId];
	Matrix3x3 m1 = m3Input[gId];
	Matrix4x4 m2 = m4Input[gId];

	Matrix2x2 m3 = fInput + gId * 4;
	Matrix3x3 m4 = fInput + gId * 4;
	Matrix4x4 m5 = fInput + gId * 4;

	// Do some Operation (to Prevent optimization)
	m0 += Matrix2x2(0.0f);
	m1 += Matrix3x3(0.0f);
	m2 += Matrix4x4(0.0f);

	m3 += Matrix2x2(0.0f);
	m4 += Matrix3x3(0.0f);
	m5 += Matrix4x4(0.0f);

	// Store
	m2Output[gId] = m0;
	m3Output[gId] = m1;
	m4Output[gId] = m2;

	reinterpret_cast<Matrix2x2*>(fM2Output)[gId] = m3;
	reinterpret_cast<Matrix3x3*>(fM3Output)[gId] = m4;
	reinterpret_cast<Matrix4x4*>(fM4Output)[gId] = m5;
}

TEST(MatrixGPU, Construction)
{

}
