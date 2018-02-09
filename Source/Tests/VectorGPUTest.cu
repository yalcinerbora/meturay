#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Vector.h"

#include <cuda_runtime.h>

using ::testing::ElementsAre;
using ::testing::FloatEq;

static constexpr unsigned int THREAD_COUNT = 50'000'000;

__global__ void KGlobalLoadStore(const Vector2* v2Input,
								 const Vector3* v3Input,
								 const Vector4* v4Input,
								 const float* fInput,

								 Vector2* v2Output,
								 Vector3* v3Output,
								 Vector4* v4Output,
								 float* fV2Output,
								 float* fV3Output,
								 float* fV4Output)
{
	unsigned int gId = threadIdx.x + blockIdx.x * blockDim.x;

	// Testing Direct Load and Store
	Vector2 v0 = v2Input[gId];
	Vector3 v1 = v3Input[gId];
	Vector4 v2 = v4Input[gId];

	Vector2 v3 = fInput[gId * 4];
	Vector3 v4 = fInput[gId * 4];
	Vector4 v5 = fInput[gId * 4];
	//
	v2Output[gId] = v0;
	v3Output[gId] = v1;
	v4Output[gId] = v2;

	reinterpret_cast<Vector2*>(fV2Output)[gId] = v3;
	reinterpret_cast<Vector3*>(fV3Output)[gId] = v4;
	reinterpret_cast<Vector4*>(fV4Output)[gId] = v5;
}

__global__ void KConstruction(Vector2* v2Array, 
							  Vector4* v4Array,

							  Vector4* results)
{
	//Vector3 vec(1.0f, 1u, 2.0f);
}

__global__ void KOperators(Vector4* arrayOfVecs,
						   Vector4* results)
{
	//Vector3 vec(1.0f, 1u, 2.0f);
}

__global__ void KFunctions(Vector4* arrayOfVecs,
						   Vector4* results)
{
	//Vector3 vec(1.0f, 1u, 2.0f);
}

TEST(VectorGPU, GlobalLoadStore)
{

}

TEST(VectorGPU, Construction)
{


}

TEST(VectorGPU, Operators)
{}

TEST(VectorGPU, Functions)
{}