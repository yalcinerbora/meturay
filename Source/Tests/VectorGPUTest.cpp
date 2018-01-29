#include <gtest/gtest.h>

#include "RayLib/Vector.h"


#include <cuda_runtime.h>

__global__ void Kernel(Vector3* arrayOfVecs)
{
	Vector3 vec(1.0f, 1u, 2.0f);
}

TEST(VectorGPU, Construction)
{

	const float test[] = { 1.0f, 2.0f, 3.0f };

	Vector3 vec(1.0f, 1u, 2.0f);
}