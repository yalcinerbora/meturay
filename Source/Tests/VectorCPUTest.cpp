#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Vector.h"

using ::testing::ElementsAre;

TEST(VectorCPU, Construction)
{
	const float dataArray[] = { 1.0f, 2.0f, 3.0f, 4.0f};
	const float dataArraySmall[] = { 1.0f, 2.0f, 3.0f};
	const float dataArrayLarge[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

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

	// Copy Assignment (default)
	Vector4 vec8;
	vec8 = vec7;

	//	
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec0),
								  static_cast<const float*>(vec0) + 4),
				ElementsAre(0.0f, 0.0f, 0.0f, 0.0f));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec1),
								   static_cast<const float*>(vec1) + 4),
				ElementsAre(1.0f, 1.0f, 1.0f, 1.0f));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec2),
								   static_cast<const float*>(vec2) + 4),
				ElementsAre(1.0f, 1.0f, 2.0f, 3.0f));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec3),
								   static_cast<const float*>(vec3) + 4),
				ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec4),
								   static_cast<const float*>(vec4) + 4),
				ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));


	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec5),
								   static_cast<const float*>(vec5) + 4),
				ElementsAre(1.0f, 2.0f, 0.0f, 0.0f));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec6),
								   static_cast<const float*>(vec6) + 4),
				ElementsAre(1.0f, 2.0f, 3.0f, 0.0f));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec7),
								   static_cast<const float*>(vec7) + 4),
				ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(vec8),
								   static_cast<const float*>(vec8) + 4),
				ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
}

TEST(VectorCPU, Operators)
{

}


TEST(VectorCPU, Functions)
{

}