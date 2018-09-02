#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Vector.h"

using ::testing::ElementsAre;
using ::testing::FloatEq;

TEST(VectorCPU, Construction)
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

	//	
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec0),
								  static_cast<const float*>(vec0) + 4),
				ElementsAre(FloatEq(0.0f), FloatEq(0.0f), FloatEq(0.0f), FloatEq(0.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec1),
								   static_cast<const float*>(vec1) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec2),
								   static_cast<const float*>(vec2) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec3),
								   static_cast<const float*>(vec3) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec4),
								   static_cast<const float*>(vec4) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));


	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec5),
								   static_cast<const float*>(vec5) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(0.0f), FloatEq(0.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec6),
								   static_cast<const float*>(vec6) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(0.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec7),
								   static_cast<const float*>(vec7) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec8),
								   static_cast<const float*>(vec8) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec9),
								   static_cast<const float*>(vec9) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec10),
								   static_cast<const float*>(vec10) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f), FloatEq(4.0f)));
}

TEST(VectorCPU, Operators)
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
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v0),
								   static_cast<const float*>(v0) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(3.0f), FloatEq(3.0f), FloatEq(3.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v1),
								   static_cast<const float*>(v1) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v2),
								   static_cast<const float*>(v2) + 4),
				ElementsAre(FloatEq(4.0f), FloatEq(8.0f), FloatEq(12.0f), FloatEq(16.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v3),
								   static_cast<const float*>(v3) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(0.5f), FloatEq(0.3333333f), FloatEq(0.25f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v4a),
								   static_cast<const float*>(v4a) + 4),
				ElementsAre(FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v4b),
								   static_cast<const float*>(v4b) + 4),
				ElementsAre(FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v5),
								   static_cast<const float*>(v5) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f), FloatEq(1.0f)));

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
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v6),
								   static_cast<const float*>(v6) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(3.0f), FloatEq(3.0f), FloatEq(3.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v7),
								   static_cast<const float*>(v7) + 4),
				ElementsAre(FloatEq(-1.0f), FloatEq(-1.0f), FloatEq(-1.0f), FloatEq(-1.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v8),
								   static_cast<const float*>(v8) + 4),
				ElementsAre(FloatEq(4.0f), FloatEq(8.0f), FloatEq(12.0f), FloatEq(16.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v9),
								   static_cast<const float*>(v9) + 4),
				ElementsAre(FloatEq(2.0f), FloatEq(2.0f), FloatEq(2.0f), FloatEq(2.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v10),
								   static_cast<const float*>(v10) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(0.5f), FloatEq(0.3333333f), FloatEq(0.25f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v11),
								   static_cast<const float*>(v11) + 4),
				ElementsAre(FloatEq(0.5f), FloatEq(0.5f), FloatEq(0.5f), FloatEq(0.5f)));	
}

TEST(VectorCPU, Functions1)
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
	//
	EXPECT_FLOAT_EQ(8.0f, v0);
	EXPECT_FLOAT_EQ(5.0f, v1);
	EXPECT_FLOAT_EQ(25.0f, v2);
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v3),
								   static_cast<const float*>(v3) + 4),
				ElementsAre(FloatEq(0.18257418f), FloatEq(0.36514837f), FloatEq(0.54772255f), FloatEq(0.73029674f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v4),
								   static_cast<const float*>(v4) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v5),
								   static_cast<const float*>(v5) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));

	// Selfs
	Vector4 v6 = c;
	v6.NormalizeSelf();

	Vector4 v7 = c;
	v7.ClampSelf(Vector4(3.0f), Vector4(4.0f));

	Vector4 v8 = c;
	v8.ClampSelf(3.0f, 4.0f);
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v6),
								   static_cast<const float*>(v6) + 4),
				ElementsAre(FloatEq(0.18257418f), FloatEq(0.36514837f), FloatEq(0.54772255f), FloatEq(0.73029674f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v7),
								   static_cast<const float*>(v7) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v8),
								   static_cast<const float*>(v8) + 4),
				ElementsAre(FloatEq(3.0f), FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));

	// Cross Product (3D Vector Special)
	Vector3 v9 = Cross(e, f);
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v9),
								   static_cast<const float*>(v9) + 3),
				ElementsAre(FloatEq(-1.0f), FloatEq(2.0f), FloatEq(-1.0f)));
}

TEST(VectorCPU, Functions2)
{
	Vector4 b(-2.12f, -2.5f, 2.60f, 2.3f);
	Vector4 c(-1.0f, 0.0f, -0.0f, 1.0f);

	Vector4 v0 = c.Abs();
	Vector4 v1 = b.Round<float>();
	Vector4 v2 = b.Floor<float>();
	Vector4 v3 = b.Ceil<float>();
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v0),
								   static_cast<const float*>(v0) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(0.0f), FloatEq(-0.0f), FloatEq(1.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v1),
								   static_cast<const float*>(v1) + 4),
				ElementsAre(FloatEq(-2.0f), FloatEq(-3.0f), FloatEq(3.0f), FloatEq(2.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v2),
								   static_cast<const float*>(v2) + 4),
				ElementsAre(FloatEq(-3.0f), FloatEq(-3.0f), FloatEq(2.0f), FloatEq(2.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v3),
								   static_cast<const float*>(v3) + 4),
				ElementsAre(FloatEq(-2.0f), FloatEq(-2.0f), FloatEq(3.0f), FloatEq(3.0f)));
	
	// Self Equavilents
	Vector4 v4 = c;
	v4.AbsSelf();
	Vector4 v5 = b;
	v5.RoundSelf();
	Vector4 v6 = b;
	v6.FloorSelf();
	Vector4 v7 = b;
	v7.CeilSelf();
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v4),
								   static_cast<const float*>(v4) + 4),
				ElementsAre(FloatEq(1.0f), FloatEq(0.0f), FloatEq(-0.0f), FloatEq(1.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v5),
								   static_cast<const float*>(v5) + 4),
				ElementsAre(FloatEq(-2.0f), FloatEq(-3.0f), FloatEq(3.0f), FloatEq(2.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v6),
								   static_cast<const float*>(v6) + 4),
				ElementsAre(FloatEq(-3.0f), FloatEq(-3.0f), FloatEq(2.0f), FloatEq(2.0f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v7),
								   static_cast<const float*>(v7) + 4),
				ElementsAre(FloatEq(-2.0f), FloatEq(-2.0f), FloatEq(3.0f), FloatEq(3.0f)));
}

TEST(VectorCPU, Functions3)
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
	//
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v0),
								   static_cast<const float*>(v0) + 4),
				ElementsAre(FloatEq(2.12f), FloatEq(2.5f), FloatEq(2.6f), FloatEq(2.3f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v1),
								   static_cast<const float*>(v1) + 4),
				ElementsAre(FloatEq(999.0f), FloatEq(999.0f), FloatEq(999.0f), FloatEq(999.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v2),
								   static_cast<const float*>(v2) + 4),
				ElementsAre(FloatEq(-2.12f), FloatEq(-2.5f), FloatEq(-2.6f), FloatEq(-2.3f)));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v3),
								   static_cast<const float*>(v3) + 4),
				ElementsAre(FloatEq(-999.0f), FloatEq(-999.0f), FloatEq(-999.0f), FloatEq(-999.0f)));

	EXPECT_THAT(std::vector<float>(static_cast<const float*>(v4),
								   static_cast<const float*>(v4) + 4),
				ElementsAre(FloatEq(0.5f), FloatEq(0.5f), FloatEq(0.5f), FloatEq(0.5f)));
}