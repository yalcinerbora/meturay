#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Quaternion.h"

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::FloatNear;


TEST(QuaternionCPU, Constructors)
{

    //

}

TEST(QuaternionCPU, Space)
{
    QuatF q0;
    Vector3 result0;
    TransformGen::Space(q0, XAxis, YAxis, ZAxis);
    EXPECT_FLOAT_EQ(IdentityQuatF[0], q0[0]);
    EXPECT_FLOAT_EQ(IdentityQuatF[1], q0[1]);
    EXPECT_FLOAT_EQ(IdentityQuatF[2], q0[2]);
    EXPECT_FLOAT_EQ(IdentityQuatF[3], q0[3]);
    result0 = q0.ApplyRotation(XAxis);
    EXPECT_FLOAT_EQ(XAxis[0], result0[0]);
    EXPECT_FLOAT_EQ(XAxis[1], result0[1]);
    EXPECT_FLOAT_EQ(XAxis[2], result0[2]);
    result0 = q0.ApplyRotation(YAxis);
    EXPECT_FLOAT_EQ(YAxis[0], result0[0]);
    EXPECT_FLOAT_EQ(YAxis[1], result0[1]);
    EXPECT_FLOAT_EQ(YAxis[2], result0[2]);
    result0 = q0.ApplyRotation(ZAxis);
    EXPECT_FLOAT_EQ(ZAxis[0], result0[0]);
    EXPECT_FLOAT_EQ(ZAxis[1], result0[1]);
    EXPECT_FLOAT_EQ(ZAxis[2], result0[2]);

    QuatF q1;
    Vector3 result1;
    QuatF resultYZX = (QuatF(-90.0f * MathConstants::DegToRadCoef, YAxis) * 
                       QuatF(-90.0f * MathConstants::DegToRadCoef, XAxis));
    resultYZX.ConjugateSelf();

    TransformGen::Space(q1, YAxis, ZAxis, XAxis);
    EXPECT_FLOAT_EQ(resultYZX[0], q1[0]);
    EXPECT_FLOAT_EQ(resultYZX[1], q1[1]);
    EXPECT_FLOAT_EQ(resultYZX[2], q1[2]);
    EXPECT_FLOAT_EQ(resultYZX[3], q1[3]);
    result1 = q1.ApplyRotation(XAxis);
    result0 = resultYZX.ApplyRotation(XAxis);
    EXPECT_FLOAT_EQ(ZAxis[0], result1[0]);
    EXPECT_FLOAT_EQ(ZAxis[1], result1[1]);
    EXPECT_FLOAT_EQ(ZAxis[2], result1[2]);
    result1 = q1.ApplyRotation(YAxis);
    result0 = resultYZX.ApplyRotation(YAxis);
    EXPECT_FLOAT_EQ(XAxis[0], result1[0]);
    EXPECT_FLOAT_EQ(XAxis[1], result1[1]);
    EXPECT_FLOAT_EQ(XAxis[2], result1[2]);
    result1 = q1.ApplyRotation(ZAxis);
    result0 = resultYZX.ApplyRotation(ZAxis);
    EXPECT_FLOAT_EQ(YAxis[0], result1[0]);
    EXPECT_FLOAT_EQ(YAxis[1], result1[1]);
    EXPECT_FLOAT_EQ(YAxis[2], result1[2]);
    
    QuatF q2;
    Vector3 result2;
    QuatF resultZXY = (QuatF(90.0f * MathConstants::DegToRadCoef, XAxis) *
                       QuatF(90.0f * MathConstants::DegToRadCoef, ZAxis));
    resultZXY.ConjugateSelf();
    TransformGen::Space(q2, ZAxis, XAxis, YAxis);
    EXPECT_FLOAT_EQ(resultZXY[0], q2[0]);
    EXPECT_FLOAT_EQ(resultZXY[1], q2[1]);
    EXPECT_FLOAT_EQ(resultZXY[2], q2[2]);
    EXPECT_FLOAT_EQ(resultZXY[3], q2[3]);
    result2 = q2.ApplyRotation(XAxis);
    result1 = resultZXY.ApplyRotation(XAxis);
    EXPECT_FLOAT_EQ(YAxis[0], result2[0]);
    EXPECT_FLOAT_EQ(YAxis[1], result2[1]);
    EXPECT_FLOAT_EQ(YAxis[2], result2[2]);
    result2 = q2.ApplyRotation(YAxis);
    result1 = resultZXY.ApplyRotation(YAxis);
    EXPECT_FLOAT_EQ(ZAxis[0], result2[0]);
    EXPECT_FLOAT_EQ(ZAxis[1], result2[1]);
    EXPECT_FLOAT_EQ(ZAxis[2], result2[2]);
    result2 = q2.ApplyRotation(ZAxis);
    result1 = resultZXY.ApplyRotation(ZAxis);
    EXPECT_FLOAT_EQ(XAxis[0], result2[0]);
    EXPECT_FLOAT_EQ(XAxis[1], result2[1]);
    EXPECT_FLOAT_EQ(XAxis[2], result2[2]);

    // Left Handed Coord
    QuatF q3;
    EXPECT_DEBUG_DEATH(TransformGen::Space(q3, ZAxis, YAxis, XAxis), ".*");
    EXPECT_FLOAT_EQ(IdentityQuatF[0], q0[0]);
    EXPECT_FLOAT_EQ(IdentityQuatF[1], q0[1]);
    EXPECT_FLOAT_EQ(IdentityQuatF[2], q0[2]);
    EXPECT_FLOAT_EQ(IdentityQuatF[3], q0[3]);
}