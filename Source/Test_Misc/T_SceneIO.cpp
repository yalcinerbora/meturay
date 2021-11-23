#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <sstream>
#include <string>
#include "RayLib/SceneIO.h"
#include "RayLib/Log.h"

// Common
// String Test
const std::string jsonString = "{\"v\":\"This is string.\"}";
// Number Test
const std::string jsonDouble = "{\"v\":0.0}";
const std::string jsonInt = "{\"v\":1}";
// Quat Test
const std::string jsonQuat = "{\"v\":[1.0, 0.0, 0.0, 0.0]}";
const std::string jsonQuatLess = "{\"v\":[1.0, 0.0, 0.0]}";
const std::string jsonQuatMore = "{\"v\":[1.0, 0.0, 0.0, 0.0, 0.0]}";

static nlohmann::json LoadJson(const std::string&s)
{
    std::istringstream str(s);
    nlohmann::json jsonFile;
    str >> jsonFile;
    return nlohmann::json(jsonFile["v"]);
}

TEST(SceneIO, String)
{
    EXPECT_STREQ(SceneIO::LoadString(LoadJson(jsonString)).c_str(), "This is string.");
    EXPECT_THROW(SceneIO::LoadString(LoadJson(jsonQuat)), SceneException);
    EXPECT_THROW(SceneIO::LoadString(LoadJson(jsonDouble)), SceneException);
}

TEST(SceneIO, Number)
{
    EXPECT_EQ(SceneIO::LoadNumber<int>(LoadJson(jsonDouble)), 0);
    EXPECT_EQ(SceneIO::LoadNumber<unsigned int>(LoadJson(jsonDouble)), 0u);
    EXPECT_FLOAT_EQ(SceneIO::LoadNumber<float>(LoadJson(jsonDouble)), 0.0f);
    EXPECT_DOUBLE_EQ(SceneIO::LoadNumber<double>(LoadJson(jsonDouble)), 0.0);

    EXPECT_EQ(SceneIO::LoadNumber<int>(LoadJson(jsonInt)), 1);
    EXPECT_EQ(SceneIO::LoadNumber<unsigned int>(LoadJson(jsonInt)), 1u);
    EXPECT_FLOAT_EQ(SceneIO::LoadNumber<float>(LoadJson(jsonInt)), 1.0f);
    EXPECT_DOUBLE_EQ(SceneIO::LoadNumber<double>(LoadJson(jsonInt)), 1.0);

    EXPECT_THROW(SceneIO::LoadNumber<float>(LoadJson(jsonQuat)), SceneException);
    EXPECT_THROW(SceneIO::LoadNumber<float>(LoadJson(jsonString)), SceneException);
}

TEST(SceneIO, Quaternion)
{
    EXPECT_EQ(SceneIO::LoadQuaternion<float>(LoadJson(jsonQuat)),
              QuatF(1.0f, 0.0f, 0.0f, 0.0f));

    EXPECT_THROW(SceneIO::LoadQuaternion<float>(LoadJson(jsonQuatLess)), SceneException);
    EXPECT_THROW(SceneIO::LoadQuaternion<float>(LoadJson(jsonQuatMore)), SceneException);
    EXPECT_THROW(SceneIO::LoadQuaternion<float>(LoadJson(jsonString)), SceneException);
    EXPECT_THROW(SceneIO::LoadQuaternion<float>(LoadJson(jsonDouble)), SceneException);
}

TEST(SceneIO, Vector)
{
    const std::string jsonVec4 = "{\"v\":[1.0, 0.0, 0.0, 0.0]}";
    const std::string jsonVec4Less = "{\"v\":[1.0, 0.0, 0.0]}";
    const std::string jsonVec4More = "{\"v\":[1.0, 0.0, 0.0, 0.0, 0.0]}";

    EXPECT_EQ((SceneIO::LoadVector<4, float>(LoadJson(jsonVec4))),
              Vector4f(1.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_THROW((SceneIO::LoadVector<4,float>(LoadJson(jsonVec4Less))), SceneException);
    EXPECT_THROW((SceneIO::LoadVector<4,float>(LoadJson(jsonVec4More))), SceneException);
    EXPECT_THROW((SceneIO::LoadVector<4,float>(LoadJson(jsonString))), SceneException);
    EXPECT_THROW((SceneIO::LoadVector<4,float>(LoadJson(jsonDouble))), SceneException);
}

TEST(SceneIO, Matrix)
{
    const std::string jsonMat4 = "{\"v\":[1.0, 0.0, 0.0, 0.0,"
                                     "0.0, 1.0, 0.0, 0.0,"
                                     "0.0, 0.0, 1.0, 0.0,"
                                     "0.0, 0.0, 0.0, 1.0]}";
    const std::string jsonMat4Less = "{\"v\":[1.0, 0.0, 0.0]}";
    const std::string jsonMat4More = "{\"v\":[1.0, 0.0, 0.0, 0.0,"
                                             "0.0, 1.0, 0.0, 0.0,"
                                             "0.0, 0.0, 1.0, 0.0,"
                                             "0.0, 0.0, 0.0, 1.0, 0.0]}";

    EXPECT_EQ((SceneIO::LoadMatrix<4, float>(LoadJson(jsonMat4))),
              Matrix4x4f(1.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 1.0f));
    EXPECT_THROW((SceneIO::LoadMatrix<4, float>(LoadJson(jsonMat4Less))), SceneException);
    EXPECT_THROW((SceneIO::LoadMatrix<4, float>(LoadJson(jsonMat4More))), SceneException);
    EXPECT_THROW((SceneIO::LoadMatrix<4, float>(LoadJson(jsonString))), SceneException);
    EXPECT_THROW((SceneIO::LoadMatrix<4, float>(LoadJson(jsonDouble))), SceneException);
}

TEST(SceneIO, AnimCheck)
{
    const std::string jsonAnim = "{\"v\":\"_test.manim\"}";
    const std::string jsonAnimWrongExt = "{\"v\":\"_test.json\"}";
    const std::string jsonAnimNoExt = "{\"v\":\"_test\"}";

    // All valid but file not found
    EXPECT_THROW((SceneIO::LoadMatrix<4, float>(LoadJson(jsonAnim))), SceneException);
    EXPECT_THROW((SceneIO::LoadMatrix<4, float>(LoadJson(jsonAnimWrongExt))), SceneException);
    EXPECT_THROW((SceneIO::LoadMatrix<4, float>(LoadJson(jsonAnimNoExt))), SceneException);

    // Actual read etc...
    METU_LOG("Animation scene system has not yet been implemented.");
    FAIL();
}