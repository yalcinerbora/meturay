#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Matrix.h"
//#include "RayLib/Constans.h"

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::FloatNear;



static Matrix4x4 testMatrix(1.0f, 5.0f, 9.0f, 13.0f,
							2.0f, 6.0f, 10.0f, 14.0f,
							3.0f, 7.0f, 11.0f, 15.0f,
							4.0f, 8.0f, 12.0f, 16.0f);
static Matrix4x4 testMatrix2 = Indentity4x4;

static float vector4[] = {99.0f, 88.0f, 77.0f, 66.0f};

// Get Set Operations
TEST(MatrixCPU, AccessorsMutators)
{
	// Accessors Also Tests Constructors
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(testMatrix),
								   static_cast<const float*>(testMatrix) + 16),
				ElementsAreArray({FloatEq(1.0), FloatEq(5.0f), FloatEq(9.0f), FloatEq(13.0f),
								  FloatEq(2.0), FloatEq(6.0f), FloatEq(10.0f), FloatEq(14.0f),
								  FloatEq(3.0), FloatEq(7.0f), FloatEq(11.0f), FloatEq(15.0f),
								  FloatEq(4.0), FloatEq(8.0f), FloatEq(12.0f), FloatEq(16.0f)}));

	//EXPECT_FLOAT_EQ(1.0f, testMatrix(0, 0));
	//EXPECT_FLOAT_EQ(2.0f, testMatrix(0, 1));
	//EXPECT_FLOAT_EQ(3.0f, testMatrix(0, 2));
	//EXPECT_FLOAT_EQ(4.0f, testMatrix(0, 3));
	//EXPECT_FLOAT_EQ(5.0f, testMatrix(1, 0));
	//EXPECT_FLOAT_EQ(6.0f, testMatrix(1, 1));
	//EXPECT_FLOAT_EQ(7.0f, testMatrix(1, 2));
	//EXPECT_FLOAT_EQ(8.0f, testMatrix(1, 3));
	//EXPECT_FLOAT_EQ(9.0f, testMatrix(2, 0));
	//EXPECT_FLOAT_EQ(10.0f, testMatrix(2, 1));
	//EXPECT_FLOAT_EQ(11.0f, testMatrix(2, 2));
	//EXPECT_FLOAT_EQ(12.0f, testMatrix(2, 3));
	//EXPECT_FLOAT_EQ(13.0f, testMatrix(3, 0));
	//EXPECT_FLOAT_EQ(14.0f, testMatrix(3, 1));
	//EXPECT_FLOAT_EQ(15.0f, testMatrix(3, 2));
	//EXPECT_FLOAT_EQ(16.0f, testMatrix(3, 3));

	// Testing Assignment
	EXPECT_FLOAT_EQ(1.0f, testMatrix2(0, 0));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(0, 1));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(0, 2));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(0, 3));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(1, 0));
	EXPECT_FLOAT_EQ(1.0f, testMatrix2(1, 1));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(1, 2));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(1, 3));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(2, 0));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(2, 1));
	EXPECT_FLOAT_EQ(1.0f, testMatrix2(2, 2));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(2, 3));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(3, 0));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(3, 1));
	EXPECT_FLOAT_EQ(0.0f, testMatrix2(3, 2));
	EXPECT_FLOAT_EQ(1.0f, testMatrix2(3, 3));

	// Testing cp Constructor
	Matrix4x4 copyMatrix(testMatrix);
	EXPECT_FLOAT_EQ(1.0f, copyMatrix(0, 0));
	EXPECT_FLOAT_EQ(2.0f, copyMatrix(0, 1));
	EXPECT_FLOAT_EQ(3.0f, copyMatrix(0, 2));
	EXPECT_FLOAT_EQ(4.0f, copyMatrix(0, 3));
	EXPECT_FLOAT_EQ(5.0f, copyMatrix(1, 0));
	EXPECT_FLOAT_EQ(6.0f, copyMatrix(1, 1));
	EXPECT_FLOAT_EQ(7.0f, copyMatrix(1, 2));
	EXPECT_FLOAT_EQ(8.0f, copyMatrix(1, 3));
	EXPECT_FLOAT_EQ(9.0f, copyMatrix(2, 0));
	EXPECT_FLOAT_EQ(10.0f, copyMatrix(2, 1));
	EXPECT_FLOAT_EQ(11.0f, copyMatrix(2, 2));
	EXPECT_FLOAT_EQ(12.0f, copyMatrix(2, 3));
	EXPECT_FLOAT_EQ(13.0f, copyMatrix(3, 0));
	EXPECT_FLOAT_EQ(14.0f, copyMatrix(3, 1));
	EXPECT_FLOAT_EQ(15.0f, copyMatrix(3, 2));
	EXPECT_FLOAT_EQ(16.0f, copyMatrix(3, 3));

	const float* testData = static_cast<const float*>(testMatrix);
	EXPECT_FLOAT_EQ(1.0f, testData[0]);
	EXPECT_FLOAT_EQ(5.0f, testData[1]);
	EXPECT_FLOAT_EQ(9.0f, testData[2]);
	EXPECT_FLOAT_EQ(13.0f, testData[3]);
	EXPECT_FLOAT_EQ(2.0f, testData[4]);
	EXPECT_FLOAT_EQ(6.0f, testData[5]);
	EXPECT_FLOAT_EQ(10.0f, testData[6]);
	EXPECT_FLOAT_EQ(14.0f, testData[7]);
	EXPECT_FLOAT_EQ(3.0f, testData[8]);
	EXPECT_FLOAT_EQ(7.0f, testData[9]);
	EXPECT_FLOAT_EQ(11.0f, testData[10]);
	EXPECT_FLOAT_EQ(15.0f, testData[11]);
	EXPECT_FLOAT_EQ(4.0f, testData[12]);
	EXPECT_FLOAT_EQ(8.0f, testData[13]);
	EXPECT_FLOAT_EQ(12.0f, testData[14]);
	EXPECT_FLOAT_EQ(16.0f, testData[15]);
}

TEST(MatrixCPU, Operators)
{
	// Equality Operators
	EXPECT_TRUE(testMatrix == testMatrix);
	EXPECT_FALSE(testMatrix == testMatrix2);
	EXPECT_TRUE(testMatrix != testMatrix2);
	EXPECT_FALSE(testMatrix != testMatrix);

	// Std Operators
	EXPECT_EQ(Matrix4x4(2.0f, 5.0f, 9.0f, 13.0f,
						2.0f, 7.0f, 10.0f, 14.0f,
						3.0f, 7.0f, 12.0f, 15.0f,
						4.0f, 8.0f, 12.0f, 17.0f), testMatrix + testMatrix2);

	EXPECT_EQ(Matrix4x4(0.0f, 5.0f, 9.0f, 13.0f,
						2.0f, 5.0f, 10.0f, 14.0f,
						3.0f, 7.0f, 10.0f, 15.0f,
						4.0f, 8.0f, 12.0f, 15.0f), testMatrix - testMatrix2);

	EXPECT_EQ(Matrix4x4(-1.0f, -5.0f, -9.0f, -13.0f,
						-2.0f, -6.0f, -10.0f, -14.0f,
						-3.0f, -7.0f, -11.0f, -15.0f,
						-4.0f, -8.0f, -12.0f, -16.0f), -testMatrix);

	EXPECT_EQ(Matrix4x4(1.0f, 5.0f, 9.0f, 13.0f,
						2.0f, 6.0f, 10.0f, 14.0f,
						3.0f, 7.0f, 11.0f, 15.0f,
						4.0f, 8.0f, 12.0f, 16.0f), testMatrix / 1.0f);

	EXPECT_EQ(Matrix4x4(1.0f, 5.0f, 9.0f, 13.0f,
						2.0f, 6.0f, 10.0f, 14.0f,
						3.0f, 7.0f, 11.0f, 15.0f,
						4.0f, 8.0f, 12.0f, 16.0f), testMatrix * 1.0f);

	EXPECT_EQ(Matrix4x4(1.0f, 5.0f, 9.0f, 13.0f,
						2.0f, 6.0f, 10.0f, 14.0f,
						3.0f, 7.0f, 11.0f, 15.0f,
						4.0f, 8.0f, 12.0f, 16.0f), 1.0f * testMatrix);

	// Matrix Matrix Mult
	Matrix4x4 multMatrix(1.0f, 5.0f, 9.0f, 13.0f,
						 2.0f, 6.0f, 10.0f, 14.0f,
						 3.0f, 7.0f, 11.0f, 15.0f,
						 4.0f, 8.0f, 12.0f, 16.0f);
	EXPECT_EQ(Matrix4x4(90.0f, 202.0f, 314.0f, 426.0f,
						100.0f, 228.0f, 356.0f, 484.0f,
						110.0f, 254.0f, 398.0f, 542.0f,
						120.0f, 280.0f, 440.0f, 600.0f), multMatrix * multMatrix);

	// Matrix Vector Mult
	EXPECT_EQ(Vector3(14.0f, 38.0f, 62.0f), testMatrix * Vector3(1.0f, 2.0f, 3.0f));
	EXPECT_EQ(Vector4(30.0f, 70.0f, 110.0f, 150.0f), testMatrix * Vector4(1.0f, 2.0f, 3.0f, 4.0f));

	// Operate and Assign Operators
	Matrix4x4 leftCopy(testMatrix);
	leftCopy *= testMatrix;
	EXPECT_EQ(Matrix4x4(90.0f, 202.0f, 314.0f, 426.0f,
						100.0f, 228.0f, 356.0f, 484.0f,
						110.0f, 254.0f, 398.0f, 542.0f,
						120.0f, 280.0f, 440.0f, 600.0f), leftCopy);

	leftCopy = testMatrix;
	leftCopy *= leftCopy;	// Self Mult Testing data dependancy
	EXPECT_EQ(Matrix4x4(90.0f, 202.0f, 314.0f, 426.0f,
						100.0f, 228.0f, 356.0f, 484.0f,
						110.0f, 254.0f, 398.0f, 542.0f,
						120.0f, 280.0f, 440.0f, 600.0f), leftCopy);

	leftCopy = testMatrix;
	leftCopy *= 2.0f;
	EXPECT_EQ(Matrix4x4(2.0f, 10.0f, 18.0f, 26.0f,
						4.0f, 12.0f, 20.0f, 28.0f,
						6.0f, 14.0f, 22.0f, 30.0f,
						8.0f, 16.0f, 24.0f, 32.0f), leftCopy);

	leftCopy = testMatrix;
	leftCopy += leftCopy;
	EXPECT_EQ(Matrix4x4(2.0f, 10.0f, 18.0f, 26.0f,
						4.0f, 12.0f, 20.0f, 28.0f,
						6.0f, 14.0f, 22.0f, 30.0f,
						8.0f, 16.0f, 24.0f, 32.0f), leftCopy);

	leftCopy = testMatrix;
	leftCopy -= leftCopy;
	EXPECT_EQ(Matrix4x4(0.0f, 0.0f, 0.0f, 0.0f,
						0.0f, 0.0f, 0.0f, 0.0f,
						0.0f, 0.0f, 0.0f, 0.0f,
						0.0f, 0.0f, 0.0f, 0.0f), leftCopy);

	leftCopy = testMatrix;
	leftCopy += leftCopy;
	leftCopy /= 2.0f;
	EXPECT_EQ(Matrix4x4(1.0f, 5.0f, 9.0f, 13.0f,
						2.0f, 6.0f, 10.0f, 14.0f,
						3.0f, 7.0f, 11.0f, 15.0f,
						4.0f, 8.0f, 12.0f, 16.0f), leftCopy);
}

TEST(MatrixCPU, LinearAlgebra)
{
	// Transpose
	EXPECT_EQ(Matrix4x4(1.0f, 2.0f, 3.0f, 4.0f,
						5.0f, 6.0f, 7.0f, 8.0f,
						9.0f, 10.0f, 11.0f, 12.0f,
						13.0f, 14.0f, 15.0f, 16.0f), testMatrix.Transpose());

	Matrix4x4 copyMatrix(testMatrix);
	EXPECT_EQ(Matrix4x4(1.0f, 2.0f, 3.0f, 4.0f,
						5.0f, 6.0f, 7.0f, 8.0f,
						9.0f, 10.0f, 11.0f, 12.0f,
						13.0f, 14.0f, 15.0f, 16.0f), copyMatrix.TransposeSelf());

	// Determinant
	EXPECT_NEAR(0.0f, copyMatrix.Determinant(), 0.0001f);

	Matrix4x4 otherMatrix(1.0f, 5.0f, 9.0f, 133.0f,
						  2.0f, 6.0f, 10.0f, 14.0f,
						  31.0f, 7.0f, 11.0f, 15.0f,
						  40.0f, 8.0f, 12.0f, 16.0f);

	EXPECT_NEAR(-9600.0f, otherMatrix.Determinant(), 0.0001f);

	// Inverse
	Matrix4x4 inverseMat = otherMatrix.Inverse();
	EXPECT_FLOAT_EQ(0.00000f, inverseMat(0, 0));
	EXPECT_FLOAT_EQ(-0.05f, inverseMat(1, 0));
	EXPECT_FLOAT_EQ(0.1f, inverseMat(2, 0));
	EXPECT_FLOAT_EQ(-0.05f, inverseMat(3, 0));
	EXPECT_FLOAT_EQ(0.0083333333333333f, inverseMat(0, 1));
	EXPECT_FLOAT_EQ(0.845f, inverseMat(1, 1));
	EXPECT_FLOAT_EQ(-4.715f, inverseMat(2, 1));
	EXPECT_FLOAT_EQ(3.611666666666667f, inverseMat(3, 1));
	EXPECT_FLOAT_EQ(-0.0166666666666667f, inverseMat(0, 2));
	EXPECT_FLOAT_EQ(-0.39f, inverseMat(1, 2));
	EXPECT_FLOAT_EQ(2.83f, inverseMat(2, 2));
	EXPECT_FLOAT_EQ(-2.173333333333333f, inverseMat(3, 2));
	EXPECT_FLOAT_EQ(0.0083333333333333f, inverseMat(0, 3));
	EXPECT_FLOAT_EQ(-0.005f, inverseMat(1, 3));
	EXPECT_FLOAT_EQ(-0.015f, inverseMat(2, 3));
	EXPECT_FLOAT_EQ(0.0116666666666667f, inverseMat(3, 3));

	// Inverse of Non Invertible
	//EXPECT_EQ(testMatrix2, testMatrix.Inverse());

	// Inverse Self
	Matrix4x4 otherMatCpy(otherMatrix);
	otherMatCpy.InverseSelf();
	EXPECT_FLOAT_EQ(0.00000f, otherMatCpy(0, 0));
	EXPECT_FLOAT_EQ(-0.05f, otherMatCpy(1, 0));
	EXPECT_FLOAT_EQ(0.1f, otherMatCpy(2, 0));
	EXPECT_FLOAT_EQ(-0.05f, otherMatCpy(3, 0));
	EXPECT_FLOAT_EQ(0.0083333333333333f, otherMatCpy(0, 1));
	EXPECT_FLOAT_EQ(0.845f, otherMatCpy(1, 1));
	EXPECT_FLOAT_EQ(-4.715f, otherMatCpy(2, 1));
	EXPECT_FLOAT_EQ(3.611666666666667f, otherMatCpy(3, 1));
	EXPECT_FLOAT_EQ(-0.0166666666666667f, otherMatCpy(0, 2));
	EXPECT_FLOAT_EQ(-0.39f, otherMatCpy(1, 2));
	EXPECT_FLOAT_EQ(2.83f, otherMatCpy(2, 2));
	EXPECT_FLOAT_EQ(-2.173333333333333f, otherMatCpy(3, 2));
	EXPECT_FLOAT_EQ(0.0083333333333333f, otherMatCpy(0, 3));
	EXPECT_FLOAT_EQ(-0.005f, otherMatCpy(1, 3));
	EXPECT_FLOAT_EQ(-0.015f, otherMatCpy(2, 3));
	EXPECT_FLOAT_EQ(0.0116666666666667f, otherMatCpy(3, 3));
}

TEST(TransformGen, Transformations)
{
	Vector3 vectorT(1.0f, 0.0f, 1.0f);
	Vector4 vectorR(0.0f, 1.0f, 0.0f, 1.0f);
	Vector4 vectorS(1.0f, 1.0f, 1.0f, 1.0f);

	// Translation
	Vector4 translatedResult = TransformGen::Translate(-vectorT) * Vector4(vectorT, 1.0f);
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(translatedResult),
								   static_cast<const float*>(translatedResult) + 4),
				ElementsAre(FloatEq(0.0), FloatEq(0.0f), FloatEq(0.0f), FloatEq(1.0f)));

	// Rotation
	Matrix4x4 rotateMatrix = TransformGen::Rotate(90.0f * MathConstants::DegToRadCoef,
												  Vector3(1.0f, 0.0f, 0.0f));
	Vector4 rotateResult = rotateMatrix * vectorR;
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(rotateResult),
								   static_cast<const float*>(rotateResult) + 4),
				ElementsAre(FloatNear(0.0f, 0.00001f), FloatNear(0.0f, 0.00001f),
							FloatNear(1.0f, 0.00001f), FloatNear(1.0f, 0.00001f)));
	//
	rotateMatrix = TransformGen::Rotate(QuatF(90.0f * MathConstants::DegToRadCoef,
											  Vector3(1.0f, 0.0f, 0.0f)));
	rotateResult = rotateMatrix * vectorR;
	EXPECT_THAT(std::vector<float>(static_cast<const float*>(rotateResult),
								   static_cast<const float*>(rotateResult) + 4),
				ElementsAre(FloatNear(0.0f, 0.00001f), FloatNear(0.0f, 0.00001f),
							FloatNear(1.0f, 0.00001f), FloatNear(1.0f, 0.00001f)));

	// Scale
	Matrix4x4 scaleMatrix = TransformGen::Scale(2.0f);
	EXPECT_EQ(Vector4(2.0f, 2.0f, 2.0f, 1.0f), scaleMatrix * vectorS);
	scaleMatrix = TransformGen::Scale(2.0f, 3.0f, 11.0f);
	EXPECT_EQ(Vector4(2.0f, 3.0f, 11.0f, 1.0f), scaleMatrix * vectorS);

	// Projection Matrices / Look At Matrix
	// Will Be Tested seperately
}
