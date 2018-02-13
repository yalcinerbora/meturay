#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Vector.h"

#include <cuda_runtime.h>

using ::testing::ElementsAre;
using ::testing::FloatEq;

//static constexpr unsigned int THREAD_COUNT = 50'000'000;

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

__global__ void KConstruction(Vector4* resuults)
{
	const float dataArray[] = {1.0f, 2.0f, 3.0f, 4.0f};
	const float dataArrayLarge[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

	Vector2 vecAssign0(1.0f, 2.0f);
	Vector3 vecAssign1(1.0f, 2.0f, 3.0f);
	Vector4 vecAssign2(1.0f, 2.0f, 3.0f, 4.0f);

	//
	Vector4 vec0;
	Vector4 vec1(1.0f);
	Vector4 vec2(1.0f, 1u, 2.0f, 3.0f);
	Vector4 vec3(dataArray);

	Vector4 vec4(dataArrayLarge);

	//
	Vector4 vec5(vecAssign0);
	Vector4 vec6(vecAssign1);

	// Copy Constructor (default)
	Vector4 vec7(vecAssign2);

	// Partial Constructor
	Vector4 vec9(vecAssign0, 3.0f, 4.0f);
	Vector4 vec10(vecAssign1, 4.0f);
}

__global__ void KOperators(Vector4* results)
{
	Vector4 a(2.0f, 2.0f, 2.0f, 2.0f);
	Vector4 b(1.0f, 1.0f, 1.0f, 1.0f);
	Vector4 c(2.0f, 4.0f, 6.0f, 8.0f);

	// Artihmetic
	Vector4 v0 = a + b;
	Vector4 v1 = a - b;
	Vector4 v2 = a * c;
	Vector4 v3 = a / c;
	Vector4 v4a = a * 2.0f;
	Vector4 v4b = 2.0f * a;
	Vector4 v5 = a / 2.0f;

	// Assignment with arithmetic
	Vector4 v6 = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
	v6 += a;

	Vector4 v7 = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
	v7 -= a;

	Vector4 v8 = Vector4(2.0f, 4.0f, 6.0f, 8.0f);
	v8 *= a;

	Vector4 v9 = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
	v9 *= 2.0f;

	Vector4 v10 = Vector4(2.0f, 2.0f, 2.0f, 2.0f);
	v10 /= c;

	Vector4 v11 = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
	v11 /= 2.0f;
}

__global__ void KFunctions1(float* floatResults,
							Vector4* results)
{
	Vector4 a(2.0f, 2.0f, 2.0f, 2.0f);
	Vector4 b(1.0f, 1.0f, 1.0f, 1.0f);
	Vector4 c(2.0f, 4.0f, 6.0f, 8.0f);
	Vector2 d(3.0f, 4.0f);
	Vector3 e(1.0f, 2.0f, 3.0f);
	Vector3 f(3.0f, 5.0f, 7.0f);

	// Non-Selfs
	float v0 = a.Dot(b);
	float v1 = d.Length();
	float v2 = d.LengthSqr();
	Vector4 v3 = c.Normalize();
	Vector4 v4 = c.Clamp(Vector4(3.0f), Vector4(4.0f));
	Vector4 v5 = c.Clamp(3.0f, 4.0f);

	// Selfs
	Vector4 v6 = c;
	v6.NormalizeSelf();

	Vector4 v7 = c;
	v7.ClampSelf(Vector4(3.0f), Vector4(4.0f));

	Vector4 v8 = c;
	v8.ClampSelf(3.0f, 4.0f);

	// Cross Product (3D Vector Special)
	Vector3 v9 = Cross(e, f);
}

__global__ void KFunctions2(Vector4* results)
{
	Vector4 b(-2.12f, -2.5f, 2.60f, 2.3f);
	Vector4 c(-1.0f, 0.0f, -0.0f, 1.0f);

	Vector4 v0 = c.Abs();
	Vector4 v1 = b.Round();
	Vector4 v2 = b.Floor();
	Vector4 v3 = b.Ceil();

	// Self Equavilents
	Vector4 v4 = c;
	v4.AbsSelf();
	Vector4 v5 = b;
	v5.RoundSelf();
	Vector4 v6 = b;
	v6.FloorSelf();
	Vector4 v7 = b;
	v7.CeilSelf();
}

__global__ void KFunctions3(Vector4* results)
{
	Vector4 a(2.12f, 2.5f, -2.60f, -2.3f);
	Vector4 b(-2.12f, -2.5f, 2.60f, 2.3f);
	Vector4 c(0.0f);
	Vector4 d(1.0f);

	Vector4 v0 = Vector4::Max(a, b);
	Vector4 v1 = Vector4::Max(a, 999.0f);

	Vector4 v2 = Vector4::Min(a, b);
	Vector4 v3 = Vector4::Min(a, -999.0f);

	Vector4 v4 = Vector4::Lerp(c, d, 0.5f);
}

TEST(VectorGPU, GlobalLoadStore)
{

}

TEST(VectorGPU, Construction)
{
	Vector4 h_data[11];
	Vector4* d_data;


	//	
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[0]),
								   static_cast<const float*>(h_data[0]) + 4),
				ElementsAre(FloatEq(0.0f), FloatEq(0.0f), FloatEq(0.0f), FloatEq(0.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[1]),
								   static_cast<const float*>(h_data[1]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[2]),
								   static_cast<const float*>(h_data[2]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[3]),
								   static_cast<const float*>(h_data[3]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[4]),
								   static_cast<const float*>(h_data[4]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));


	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[5]),
								   static_cast<const float*>(h_data[5]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(0.0f), FloatEq(0.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[6]),
								   static_cast<const float*>(h_data[6]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(0.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[7]),
								   static_cast<const float*>(h_data[7]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[8]),
								   static_cast<const float*>(h_data[8]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[9]),
								   static_cast<const float*>(h_data[9]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[10]),
								   static_cast<const float*>(h_data[10]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));
}

TEST(VectorGPU, Operators)
{
	Vector4 h_data[13];
	Vector4* d_data;

	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[0]),
								   static_cast<const float*>(h_data[0]) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(3.0f), FloatEq(3.0f), FloatEq(3.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[1]),
								   static_cast<const float*>(h_data[1]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[2]),
								   static_cast<const float*>(h_data[2]) + 4),
				ElementsAre(FloatEq(4.0f), FloatEq(8.0f), FloatEq(12.0f), FloatEq(16.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[3]),
								   static_cast<const float*>(h_data[3]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(0.5f), FloatEq(0.3333333f), FloatEq(0.25f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[4]),
								   static_cast<const float*>(h_data[4]) + 4),
				ElementsAre(FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[5]),
								   static_cast<const float*>(h_data[5]) + 4),
				ElementsAre(FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[6]),
								   static_cast<const float*>(h_data[6]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f)));
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[7]),
								   static_cast<const float*>(h_data[7]) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(3.0f), FloatEq(3.0f), FloatEq(3.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[8]),
								   static_cast<const float*>(h_data[8]) + 4),
				ElementsAre(FloatEq(-1.0f), FloatEq(-1.0f), FloatEq(-1.0f), FloatEq(-1.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[9]),
								   static_cast<const float*>(h_data[9]) + 4),
				ElementsAre(FloatEq(4.0f), FloatEq(8.0f), FloatEq(12.0f), FloatEq(16.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[10]),
								   static_cast<const float*>(h_data[10]) + 4),
				ElementsAre(FloatEq(2.0f), FloatEq(2.0f), FloatEq(2.0f), FloatEq(2.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[11]),
								   static_cast<const float*>(h_data[11]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(0.5f), FloatEq(0.3333333f), FloatEq(0.25f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[12]),
								   static_cast<const float*>(h_data[12]) + 4),
				ElementsAre(FloatEq(0.5f), FloatEq(0.5f), FloatEq(0.5f), FloatEq(0.5f)));
}

TEST(VectorGPU, Functions1)
{
	float h_data_f[3];
	float* d_data_f;

	Vector4 h_data_v4[6];
	Vector4* d_data_v4;

	Vector3 h_data_v3;
	Vector3* d_data;


	//
	EXPECT_FLOAT_EQ(8.0f, h_data_f[0]);
	EXPECT_FLOAT_EQ(5.0f, h_data_f[1]);
	EXPECT_FLOAT_EQ(25.0f, h_data_f[2]);
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data_v4[0]),
								   static_cast<const float*>(h_data_v4[0]) + 4),
				ElementsAre(FloatEq(0.18257418f), FloatEq(0.36514837f), FloatEq(0.54772255f), FloatEq(0.73029674f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data_v4[1]),
								   static_cast<const float*>(h_data_v4[1]) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data_v4[2]),
								   static_cast<const float*>(h_data_v4[2]) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data_v4[3]),
								   static_cast<const float*>(h_data_v4[3]) + 4),
				ElementsAre(FloatEq(0.18257418f), FloatEq(0.36514837f), FloatEq(0.54772255f), FloatEq(0.73029674f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data_v4[4]),
								   static_cast<const float*>(h_data_v4[4]) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data_v4[5]),
								   static_cast<const float*>(h_data_v4[5]) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data_v3),
								   static_cast<const float*>(h_data_v3) + 3),
				ElementsAre(FloatEq(-1.0f), FloatEq(2.0f), FloatEq(-1.0f)));

}

TEST(VectorGPU, Functions2)
{
	Vector4 h_data[8];
	Vector4* d_data;


	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[0]),
								   static_cast<const float*>(h_data[0]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(0.0f), FloatEq(-0.0f), FloatEq(1.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[1]),
								   static_cast<const float*>(h_data[1]) + 4),
				ElementsAre(FloatEq(-2.0f), FloatEq(-3.0f), FloatEq(3.0f), FloatEq(2.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[2]),
								   static_cast<const float*>(h_data[2]) + 4),
				ElementsAre(FloatEq(-3.0f), FloatEq(-3.0f), FloatEq(2.0f), FloatEq(2.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[3]),
								   static_cast<const float*>(h_data[3]) + 4),
				ElementsAre(FloatEq(-2.0f), FloatEq(-2.0f), FloatEq(3.0f), FloatEq(3.0f)));
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[4]),
								   static_cast<const float*>(h_data[4]) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(0.0f), FloatEq(-0.0f), FloatEq(1.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[5]),
								   static_cast<const float*>(h_data[5]) + 4),
				ElementsAre(FloatEq(-2.0f), FloatEq(-3.0f), FloatEq(3.0f), FloatEq(2.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[6]),
								   static_cast<const float*>(h_data[6]) + 4),
				ElementsAre(FloatEq(-3.0f), FloatEq(-3.0f), FloatEq(2.0f), FloatEq(2.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[7]),
								   static_cast<const float*>(h_data[7]) + 4),
				ElementsAre(FloatEq(-2.0f), FloatEq(-2.0f), FloatEq(3.0f), FloatEq(3.0f)));
}

TEST(VectorGPU, Functions3)
{
	Vector4 h_data[5];
	Vector4* d_data;


	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[0]),
								   static_cast<const float*>(h_data[0]) + 4),
				ElementsAre(FloatEq(2.12f), FloatEq(2.5f), FloatEq(2.6f), FloatEq(2.3f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[1]),
								   static_cast<const float*>(h_data[1]) + 4),
				ElementsAre(FloatEq(999.0f), FloatEq(999.0f), FloatEq(999.0f), FloatEq(999.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[2]),
								   static_cast<const float*>(h_data[2]) + 4),
				ElementsAre(FloatEq(-2.12f), FloatEq(-2.5f), FloatEq(-2.6f), FloatEq(-2.3f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[3]),
								   static_cast<const float*>(h_data[3]) + 4),
				ElementsAre(FloatEq(-999.0f), FloatEq(-999.0f), FloatEq(-999.0f), FloatEq(-999.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data[4]),
								   static_cast<const float*>(h_data[4]) + 4),
				ElementsAre(FloatEq(0.5f), FloatEq(0.5f), FloatEq(0.5f), FloatEq(0.5f)));
}