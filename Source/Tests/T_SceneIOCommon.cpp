#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/SceneIO.h"
#include "RayLib/Camera.h"

#include <fstream>
#include "RayLib/SceneIO.h"
#include "RayLib/Log.h"
#include "RayLib/SceneNodeNames.h"
#include "RayLib/StripComments.h"

static const std::string TestSceneName = "TestScenes/jsonRead.json";

static nlohmann::json ReadTestFile(const std::string& fileName = TestSceneName)
{
    std::ifstream file(fileName);  
    auto stream = Utility::StripComments(file);

    nlohmann::json jsonFile;
    stream >> jsonFile;
    return jsonFile;
}

TEST(SceneIOCommon, Camera)
{
    static constexpr CPUCamera CamResult =
    {
        0,
        CameraType::APERTURE,
        HitKey::InvalidKey,
        Vector3(0.0f, 5.0f, 0.0f),
        0.1f,
        Vector3(0.0f, 5.0f, 10.0f),
        100.0f,
        Vector3(0.0f, 1.0f, 0.0f),
        1.0f,
        Vector2(MathConstants::Pi * 0.25, MathConstants::Pi * 0.25)
    };

    nlohmann::json jsn;
    EXPECT_NO_THROW(jsn = ReadTestFile()[NodeNames::CAMERA_BASE]);
    CPUCamera camera = SceneIO::LoadCamera(jsn[0]);

    EXPECT_EQ(CamResult.mediumIndex, camera.mediumIndex);
    EXPECT_EQ(CamResult.type, camera.type);

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

TEST(SceneIOCommon, Surface)
{
    nlohmann::json jsn;
    jsn = ReadTestFile()[NodeNames::SURFACE_BASE];
    SurfaceStruct normal0 = SceneIO::LoadSurface(jsn[0]);
    EXPECT_EQ(normal0.acceleratorId, 79);
    EXPECT_EQ(normal0.transformId, 99);
    EXPECT_EQ(normal0.matPrimPairs[0].first, 100);
    EXPECT_EQ(normal0.matPrimPairs[1].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(normal0.matPrimPairs[0].second, 3);
    EXPECT_EQ(normal0.matPrimPairs[1].second, std::numeric_limits<uint32_t>::max());

    SurfaceStruct normal1 = SceneIO::LoadSurface(jsn[1]);
    EXPECT_EQ(normal1.acceleratorId, 78);
    EXPECT_EQ(normal1.transformId, 98);
    EXPECT_EQ(normal1.matPrimPairs[0].first, 101);
    EXPECT_EQ(normal1.matPrimPairs[1].first, 102);
    EXPECT_EQ(normal1.matPrimPairs[2].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(normal1.matPrimPairs[0].second, 4);
    EXPECT_EQ(normal1.matPrimPairs[1].second, 5);
    EXPECT_EQ(normal1.matPrimPairs[2].second, std::numeric_limits<uint32_t>::max());

    // Light Versions
    SurfaceStruct light0 = SceneIO::LoadSurface(jsn[2]);
    EXPECT_EQ(light0.acceleratorId, 77);
    EXPECT_EQ(light0.transformId, 97);
    EXPECT_EQ(light0.matPrimPairs[0].first, 103);
    EXPECT_EQ(light0.matPrimPairs[1].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(light0.matPrimPairs[0].second, 0x80000006);
    EXPECT_EQ(light0.matPrimPairs[1].second, std::numeric_limits<uint32_t>::max());

    SurfaceStruct light1 = SceneIO::LoadSurface(jsn[3]);
    EXPECT_EQ(light1.acceleratorId, 76);
    EXPECT_EQ(light1.transformId, 96);
    EXPECT_EQ(light1.matPrimPairs[0].first, 104);
    EXPECT_EQ(light1.matPrimPairs[1].first, 105);
    EXPECT_EQ(light1.matPrimPairs[2].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(light1.matPrimPairs[0].second, 0x80000007);
    EXPECT_EQ(light1.matPrimPairs[1].second, 0x80000008);
    EXPECT_EQ(light1.matPrimPairs[2].second, std::numeric_limits<uint32_t>::max());

    SurfaceStruct light2 = SceneIO::LoadSurface(jsn[4]);
    EXPECT_EQ(light2.acceleratorId, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(light2.transformId, 95);
    EXPECT_EQ(light2.matPrimPairs[0].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(light2.matPrimPairs[1].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(light2.matPrimPairs[0].second, 0x80000009);
    EXPECT_EQ(light2.matPrimPairs[1].second, std::numeric_limits<uint32_t>::max());

    SurfaceStruct light3 = SceneIO::LoadSurface(jsn[5]);
    EXPECT_EQ(light3.acceleratorId, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(light3.transformId, 94);
    EXPECT_EQ(light3.matPrimPairs[0].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(light3.matPrimPairs[1].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(light3.matPrimPairs[2].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(light3.matPrimPairs[0].second, 0x8000000A);
    EXPECT_EQ(light3.matPrimPairs[1].second, 0x8000000B);
    EXPECT_EQ(light3.matPrimPairs[2].second, std::numeric_limits<uint32_t>::max());

    // Hybrid
    SurfaceStruct hybrid = SceneIO::LoadSurface(jsn[6]);
    EXPECT_EQ(hybrid.acceleratorId, 73);
    EXPECT_EQ(hybrid.transformId, 93);
    EXPECT_EQ(hybrid.matPrimPairs[0].first, 2);
    EXPECT_EQ(hybrid.matPrimPairs[1].first, 6);
    EXPECT_EQ(hybrid.matPrimPairs[2].first, 7);
    EXPECT_EQ(hybrid.matPrimPairs[3].first, 8);
    EXPECT_EQ(hybrid.matPrimPairs[4].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(hybrid.matPrimPairs[0].second, 2);
    EXPECT_EQ(hybrid.matPrimPairs[1].second, 0x8000000C);
    EXPECT_EQ(hybrid.matPrimPairs[2].second, 1);
    EXPECT_EQ(hybrid.matPrimPairs[3].second, 0x8000000D);
    EXPECT_EQ(hybrid.matPrimPairs[4].second, std::numeric_limits<uint32_t>::max());

    SurfaceStruct array0 = SceneIO::LoadSurface(jsn[7]);
    EXPECT_EQ(array0.acceleratorId, 0);
    EXPECT_EQ(array0.transformId, 0);
    EXPECT_EQ(array0.matPrimPairs[0].first, 0);
    EXPECT_EQ(array0.matPrimPairs[1].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(array0.matPrimPairs[0].second, 3);
    EXPECT_EQ(array0.matPrimPairs[1].second, std::numeric_limits<uint32_t>::max());

    SurfaceStruct array1 = SceneIO::LoadSurface(jsn[8]);
    EXPECT_EQ(array1.acceleratorId, 34);
    EXPECT_EQ(array1.transformId, 33);
    EXPECT_EQ(array1.matPrimPairs[0].first, 6);
    EXPECT_EQ(array1.matPrimPairs[1].first, 8);
    EXPECT_EQ(array1.matPrimPairs[2].first, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(array1.matPrimPairs[0].second, 3);
    EXPECT_EQ(array1.matPrimPairs[1].second, 4);
    EXPECT_EQ(array1.matPrimPairs[2].first, std::numeric_limits<uint32_t>::max());

    EXPECT_THROW(SceneIO::LoadSurface(jsn[9]), SceneException);
}

TEST(SceneIOCommon, Lights)
{
    EXPECT_TRUE(false);
    //LightStruct LightPoint = {};
    //LightPoint.typeName = "point";
    //LightPoint.matId = 0;

    //LightStruct LightDirectional = {};
    //LightDirectional.typeName = "directional";
    //LightDirectional.matId = 1;
    //
    //LightStruct LightSpot = {};
    //LightSpot.typeName = "spot";
    //LightSpot.matId = 2;
    //
    //LightStruct LightRectangular = {};
    //LightRectangular.typeName = "rectangular";
    //LightRectangular.matId = 3;

    //// Read
    //LightStruct light;
    //nlohmann::json jsn = ReadTestFile()[NodeNames::LIGHT_BASE];
    //// First one is external, it should throw file not found
    //EXPECT_THROW(SceneIO::LoadLight(jsn[0]), SceneException);
    //// Point
    //light = SceneIO::LoadLight(jsn[1]);
    //EXPECT_STREQ(LightPoint.typeName.c_str(), light.typeName.c_str());
    //EXPECT_EQ(LightPoint.matId, light.matId);
    //// Directional
    //light = SceneIO::LoadLight(jsn[2]);
    //EXPECT_STREQ(LightDirectional.typeName.c_str(), light.typeName.c_str());
    //EXPECT_EQ(LightDirectional.matId, light.matId);
    //// Spot
    //light = SceneIO::LoadLight(jsn[3]);
    //EXPECT_STREQ(LightSpot.typeName.c_str(), light.typeName.c_str());
    //EXPECT_EQ(LightSpot.matId, light.matId);
    //// Rectangular
    //light = SceneIO::LoadLight(jsn[4]);
    //EXPECT_STREQ(LightRectangular.typeName.c_str(), light.typeName.c_str());
    //EXPECT_EQ(LightRectangular.matId, light.matId);
}

TEST(SceneIOCommon, Transform)
{
    GPUTransform t;
    nlohmann::json jsn;
    EXPECT_NO_THROW(jsn = ReadTestFile()[NodeNames::TRANSFORM_BASE]);
    // First one is external, it should throw file not found
    EXPECT_THROW(SceneIO::LoadTransform(jsn[0]), SceneException);
    // Matrix
    t = SceneIO::LoadTransform(jsn[1]);
    EXPECT_EQ(Indentity4x4, t);
    // TRS
    t = SceneIO::LoadTransform(jsn[2]);
    EXPECT_EQ(Indentity4x4, t);
}