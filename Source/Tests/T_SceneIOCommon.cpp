#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/SceneIO.h"
#include "RayLib/Camera.h"

#include <fstream>
#include "RayLib/SceneIO.h"
#include "RayLib/Log.h"
#include "RayLib/SceneNodeNames.h"

static const std::string TestSceneName = "TestScenes/jsonRead.json";

static nlohmann::json ReadTestFile(const std::string& fileName = TestSceneName)
{
    std::ifstream file(fileName);
    nlohmann::json jsonFile;

    std::string s;
    file >> jsonFile;
    return jsonFile;
}

TEST(SceneIOCommon, Camera)
{
    static constexpr CameraPerspective CamResult =
    {
        Vector3(0.0f, 5.0f, 0.0f),
        0.1f,
        Vector3(0.0f, 5.0f, 10.0f),
        100.0f,
        Vector3(0.0f, 1.0f, 0.0f),
        1.0f,
        Vector2(MathConstants::Pi * 0.25, MathConstants::Pi * 0.25)
    };

    nlohmann::json jsn = ReadTestFile()[NodeNames::CAMERA_BASE];
    CameraPerspective camera = SceneIO::LoadCamera(jsn[0]);
    EXPECT_FLOAT_EQ(CamResult.position[0], camera.position[0]);
    EXPECT_FLOAT_EQ(CamResult.position[1], camera.position[1]);
    EXPECT_FLOAT_EQ(CamResult.position[2], camera.position[2]);

    EXPECT_FLOAT_EQ(CamResult.up[0], camera.up[0]);
    EXPECT_FLOAT_EQ(CamResult.up[1], camera.up[1]);
    EXPECT_FLOAT_EQ(CamResult.up[2], camera.up[2]);

    EXPECT_FLOAT_EQ(CamResult.gazePoint[0], camera.gazePoint[0]);
    EXPECT_FLOAT_EQ(CamResult.gazePoint[1], camera.gazePoint[1]);
    EXPECT_FLOAT_EQ(CamResult.gazePoint[2], camera.gazePoint[2]);

    EXPECT_FLOAT_EQ(CamResult.fov[0], camera.fov[0]);
    EXPECT_FLOAT_EQ(CamResult.fov[1], camera.fov[1]);

    EXPECT_FLOAT_EQ(CamResult.nearPlane, camera.nearPlane);
    EXPECT_FLOAT_EQ(CamResult.farPlane, camera.farPlane);

    // Second one is external, it should throw file not found
    EXPECT_THROW(SceneIO::LoadCamera(jsn[1]), SceneException);
}

TEST(SceneIOCommon, Lights)
{
    LightStruct LightPoint = {};
    LightPoint.typeName = "point";
    LightPoint.matId = 0;

    LightStruct LightDirectional = {};
    LightDirectional.typeName = "directional";
    LightDirectional.matId = 1;
    
    LightStruct LightSpot = {};
    LightSpot.typeName = "spot";
    LightSpot.matId = 2;
    
    LightStruct LightRectangular = {};
    LightRectangular.typeName = "rectangular";
    LightRectangular.matId = 3;

    // Read
    LightStruct light;
    nlohmann::json jsn = ReadTestFile()[NodeNames::LIGHT_BASE];
    // First one is external, it should throw file not found
    EXPECT_THROW(SceneIO::LoadLight(jsn[0]), SceneException);
    // Point
    light = SceneIO::LoadLight(jsn[1]);
    EXPECT_STREQ(LightPoint.typeName.c_str(), light.typeName.c_str());
    EXPECT_EQ(LightPoint.matId, light.matId);
    // Directional
    light = SceneIO::LoadLight(jsn[2]);
    EXPECT_STREQ(LightDirectional.typeName.c_str(), light.typeName.c_str());
    EXPECT_EQ(LightDirectional.matId, light.matId);
    // Spot
    light = SceneIO::LoadLight(jsn[3]);
    EXPECT_STREQ(LightSpot.typeName.c_str(), light.typeName.c_str());
    EXPECT_EQ(LightSpot.matId, light.matId);
    // Rectangular
    light = SceneIO::LoadLight(jsn[4]);
    EXPECT_STREQ(LightRectangular.typeName.c_str(), light.typeName.c_str());
    EXPECT_EQ(LightRectangular.matId, light.matId);
}

TEST(SceneIOCommon, Transform)
{
    TransformStruct t;
    nlohmann::json jsn = ReadTestFile()[NodeNames::TRANSFORM_BASE];
    // First one is external, it should throw file not found
    EXPECT_THROW(SceneIO::LoadTransform(jsn[0]), SceneException);
    // Matrix
    t = SceneIO::LoadTransform(jsn[1]);
    EXPECT_EQ(Indentity4x4, t);
    // TRS
    t = SceneIO::LoadTransform(jsn[2]);
    EXPECT_EQ(Indentity4x4, t);
}