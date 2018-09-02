#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Vector.h"
#include "RayLib/DeviceMemory.h"

#include <cuda_runtime.h>

using ::testing::ElementsAre;
using ::testing::FloatEq;

__global__ void KGlobalLoadStore(const Vector2* v2Input,
								 const Vector3* v3Input,
								 const Vector4* v4Input,
								 const float* fInput,
								 const float4* f4In,

								 Vector2* v2Output,
								 Vector3* v3Output,
								 Vector4* v4Output,
								 float* fV2Output,
								 float* fV3Output,
								 float* fV4Output,
								 float4* f4Out)
{
	unsigned int gId = threadIdx.x + blockIdx.x * blockDim.x;

	// Testing Direct Load and Store
	Vector2 v0 = v2Input[gId];
	Vector3 v1 = v3Input[gId];
	Vector4 v2 = v4Input[gId];

	Vector2 v3 = fInput + gId * 4;
	Vector3 v4 = fInput + gId * 4;
	Vector4 v5 = fInput + gId * 4;
	
	float4 v6Data = f4In[gId];
	Vector4 v6(v6Data.x, v6Data.y, v6Data.z, v6Data.w);

	// Do some Operation (to Prevent optimization)
	v0 += Vector2(0.0f);
	v1 += Vector3(0.0f);
	v2 += Vector4(0.0f);

	v3 += Vector2(0.0f);
	v4 += Vector3(0.0f);
	v5 += Vector4(0.0f);

	v6 += Vector4(0.0f);

	// Store
	v2Output[gId] = v0;
	v3Output[gId] = v1;
	v4Output[gId] = v2;

	reinterpret_cast<Vector2*>(fV2Output)[gId] = v3;
	reinterpret_cast<Vector3*>(fV3Output)[gId] = v4;
	reinterpret_cast<Vector4*>(fV4Output)[gId] = v5;

	float4 v6Out;
	v6Out.x = v6[0];
	v6Out.y = v6[1];
	v6Out.z = v6[2];
	v6Out.w = v6[3];
	f4Out[gId] = v6Out;
}

__global__ void KConstruction(Vector4* results)
{
	const float dataArray[] = {1.0f, 2.0f, 3.0f, 4.0f};
	const float dataArrayLarge[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

	Vector2 vecAssign0(1.0f, 2.0f);
	Vector3 vecAssign1(1.0f, 2.0f, 3.0f);
	Vector4 vecAssign2(1.0f, 2.0f, 3.0f, 4.0f);
	//
	Vector4 vec0(0.0f);
	Vector4 vec1(1.0f);
	Vector4 vec2(1.0f, 1u, 2.0f, 3.0f);
	Vector4 vec3(dataArray);

	Vector4 vec4(dataArrayLarge);

	//
	Vector4 vec5(vecAssign0);
	Vector4 vec6(vecAssign1);

	// Copy Constructor (default)
	Vector4 vec7(vecAssign2);

	// Copy Assignment (default)
	Vector4 vec8;
	vec8 = vec7;

	// Partial Constructor
	Vector4 vec9(vecAssign0, 3.0f, 4.0f);
	Vector4 vec10(vecAssign1, 4.0f);

	results[0] = vec0;
	results[1] = vec1;
	results[2] = vec2;
	results[3] = vec3;
	results[4] = vec4;
	results[5] = vec5;
	results[6] = vec6;
	results[7] = vec7;
	results[8] = vec8;
	results[9] = vec9;
	results[10] = vec10;
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

	results[0] = v0;
	results[1] = v1;
	results[2] = v2;
	results[3] = v3;
	results[4] = v4a;
	results[5] = v4b;
	results[6] = v5;
	results[7] = v6;
	results[8] = v7;
	results[9] = v8;
	results[10] = v9;
	results[11] = v10;
	results[12] = v11;
}

__global__ void KFunctions1(float* floatResults,
							Vector3* vec3Results,
							Vector4* vec4Results)
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

	floatResults[0] = v0;
	floatResults[1] = v1;
	floatResults[2] = v2;
	vec4Results[0] = v3;
	vec4Results[1] = v4;
	vec4Results[2] = v5;
	vec4Results[3] = v6;
	vec4Results[4] = v7;
	vec4Results[5] = v8;
	vec3Results[0] = v9;
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

	results[0] = v0;
	results[1] = v1;
	results[2] = v2;
	results[3] = v3;
	results[4] = v4;
	results[5] = v5;
	results[6] = v6;
	results[7] = v7;
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

	results[0] = v0;
	results[1] = v1;
	results[2] = v2;
	results[3] = v3;
	results[4] = v4;
}

TEST(VectorGPU, GlobalLoadStore)
{
	// Global load store does not check any values
	// It is here to make compiled SASS output in order
	// to check vectors are load/stored with correct instructions 
	// (128bit loads/stores for Vector4 etc..)

	static constexpr int ElementCount = 5'000'000;

	DeviceMemory v2DataIn(ElementCount * sizeof(Vector2));
	DeviceMemory v2DataOut(ElementCount * sizeof(Vector2));

	DeviceMemory v3DataIn(ElementCount * sizeof(Vector3));
	DeviceMemory v3DataOut(ElementCount * sizeof(Vector3));

	DeviceMemory v4DataIn(ElementCount * sizeof(Vector4));
	DeviceMemory v4DataOut(ElementCount * sizeof(Vector4));

	DeviceMemory f4DataIn(ElementCount * sizeof(float4));
	DeviceMemory f4DataOut(ElementCount * sizeof(float4));

	DeviceMemory fDataIn(ElementCount * 4 * sizeof(float));
	DeviceMemory fV2DataOut(ElementCount * 4 * sizeof(float));
	DeviceMemory fV3DataOut(ElementCount * 4 * sizeof(float));
	DeviceMemory fV4DataOut(ElementCount * 4 * sizeof(float));
	
	// 
	static constexpr int BlockSize = 256;
	static constexpr int GridSize = (ElementCount + (BlockSize - 1)) / BlockSize;	
	KGlobalLoadStore<<<GridSize, BlockSize>>>(static_cast<const Vector2*>(v2DataIn),
											  static_cast<const Vector3*>(v3DataIn),
											  static_cast<const Vector4*>(v4DataIn),
											  static_cast<const float*>(fDataIn),
											  static_cast<const float4*>(f4DataIn),
											  //
											  static_cast<Vector2*>(v2DataOut),
											  static_cast<Vector3*>(v3DataOut),
											  static_cast<Vector4*>(v4DataOut),
											  static_cast<float*>(fV2DataOut),
											  static_cast<float*>(fV3DataOut),
											  static_cast<float*>(fV4DataOut),
											  static_cast<float4*>(f4DataOut));
	CUDA_KERNEL_CHECK();
}

TEST(VectorGPU, Construction)
{
	static constexpr size_t vectorCount = 11;
	static constexpr size_t vectorSize = vectorCount * sizeof(Vector4);

	DeviceMemory mem(vectorSize);
	const Vector4* h_data = static_cast<Vector4*>(mem);

	// Kernel Call
	KConstruction<<<1,1>>>(static_cast<Vector4*>(mem));
	CUDA_KERNEL_CHECK();
	CUDA_CHECK(cudaDeviceSynchronize());

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
	static constexpr size_t vectorCount = 13;
	static constexpr size_t vectorSize = vectorCount * sizeof(Vector4);

	DeviceMemory mem(vectorSize);
	const Vector4* h_data = static_cast<Vector4*>(mem);

	// Kernel Call
	KOperators<<<1,1>>>(static_cast<Vector4*>(mem));
	CUDA_KERNEL_CHECK();
	CUDA_CHECK(cudaDeviceSynchronize());

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
	static constexpr size_t fCount = 3;
	static constexpr size_t v3Count = 1;
	static constexpr size_t v4Count = 6;
	static constexpr size_t fSize = fCount * sizeof(float);
	static constexpr size_t v3Size = v3Count * sizeof(Vector3);
	static constexpr size_t v4Size = v4Count * sizeof(Vector4);

	DeviceMemory mem_f(fSize);
	DeviceMemory mem_v3(v3Size);
	DeviceMemory mem_v4(v4Size);		
	const float* h_data_f = static_cast<float*>(mem_f);
	const Vector3* h_data_v3 = static_cast<Vector3*>(mem_v3);
	const Vector4* h_data_v4 = static_cast<Vector4*>(mem_v4);

	// Kernel Call
	KFunctions1<<<1,1>>>(static_cast<float*>(mem_f),
						 static_cast<Vector3*>(mem_v3),
						 static_cast<Vector4*>(mem_v4));
	CUDA_KERNEL_CHECK();
	CUDA_CHECK(cudaDeviceSynchronize());
	
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
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(h_data_v3[0]),
								   static_cast<const float*>(h_data_v3[0]) + 3),
				ElementsAre(FloatEq(-1.0f), FloatEq(2.0f), FloatEq(-1.0f)));

}

TEST(VectorGPU, Functions2)
{
	static constexpr size_t vectorCount = 8;
	static constexpr size_t vectorSize = vectorCount * sizeof(Vector4);

	DeviceMemory mem(vectorSize);
	const Vector4* h_data = static_cast<Vector4*>(mem);

	// Kernel Call
	KFunctions2<<<1,1>>>(static_cast<Vector4*>(mem));
	CUDA_KERNEL_CHECK();
	CUDA_CHECK(cudaDeviceSynchronize());
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
	static constexpr size_t vectorCount = 5;
	static constexpr size_t vectorSize = vectorCount * sizeof(Vector4);

	DeviceMemory mem(vectorSize);
	const Vector4* h_data = static_cast<Vector4*>(mem);

	// Kernel Call
	KFunctions3<<<1,1>>>(static_cast<Vector4*>(mem));
	CUDA_KERNEL_CHECK();
	CUDA_CHECK(cudaDeviceSynchronize());
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