#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/SceneIO.h"
#include "RayLib/Camera.h"

#include <fstream>
#include "RayLib/SceneIO.h"
#include "RayLib/Log.h"

static const std::string TestSceneName = "testScene.json";

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
		Vector2(1.22173f, 0.58904f)
	};

	nlohmann::json jsn = ReadTestFile()[SceneIO::CAMERA_BASE];
	CameraPerspective camera = SceneIO::LoadCamera(jsn[0]);
	EXPECT_EQ(CamResult.position, camera.position);
	EXPECT_EQ(CamResult.up, camera.up);
	EXPECT_EQ(CamResult.gazePoint, camera.gazePoint);
	EXPECT_EQ(CamResult.fov, camera.fov);
	EXPECT_EQ(CamResult.nearPlane, camera.nearPlane);
	EXPECT_EQ(CamResult.farPlane, camera.farPlane);

	// Second one is external, it should throw file not found
	EXPECT_THROW(SceneIO::LoadCamera(jsn[1]), SceneException);
}

TEST(SceneIOCommon, Lights)
{
	LightStruct LightPoint = {};
	LightPoint.t = LightType::POINT;
	LightPoint.point.position = Vector3(0.0f);
	LightPoint.point.color = Vector3(1.0f);
	LightPoint.point.intensity = 100.0f;

	LightStruct LightDirectional = {};
	LightDirectional.t = LightType::DIRECTIONAL;
	LightDirectional.directional.direction = Vector3(0.0f);
	LightDirectional.directional.color = Vector3(1.0f);
	LightDirectional.directional.intensity = 1.0f;

	LightStruct LightSpot = {};
	LightSpot.t = LightType::SPOT;
	LightSpot.spot.position = Vector3(0.0f);
	LightSpot.spot.direction = Vector3(1.0f, -1.0f, 1.0f);
	LightSpot.spot.coverageAngle = 5;
	LightSpot.spot.falloffAngle = 4;
	LightSpot.spot.color = Vector3(1.0f);
	LightSpot.spot.intensity = 1.0f;

	LightStruct LightRectangular = {};
	LightRectangular.t = LightType::RECTANGULAR;
	LightRectangular.rectangular.position = Vector3(1.0f);
	LightRectangular.rectangular.edge0 = Vector3(2.0f, 1.0f, 1.0f);
	LightRectangular.rectangular.edge1 = Vector3(3.0f, 1.0f, 1.0f);
	LightRectangular.rectangular.red = 4.0f;
	LightRectangular.rectangular.green = 1.0f;
	LightRectangular.rectangular.blue = 1.0f;
	LightRectangular.rectangular.intensity = 1.0f;

	// Read
	LightStruct light;
	nlohmann::json jsn = ReadTestFile()[SceneIO::LIGHT_BASE];
	// First one is external, it should throw file not found
	EXPECT_THROW(SceneIO::LoadLight(jsn[0]), SceneException);
	// Point	
	light = SceneIO::LoadLight(jsn[1]);
	EXPECT_EQ(LightPoint.point.position, light.point.position);
	EXPECT_EQ(LightPoint.point.color, light.point.color);
	EXPECT_EQ(LightPoint.point.intensity, light.point.intensity);
	// Directional
	light = SceneIO::LoadLight(jsn[2]);
	EXPECT_EQ(LightDirectional.directional.direction, light.directional.direction);
	EXPECT_EQ(LightDirectional.directional.color, light.directional.color);
	EXPECT_EQ(LightDirectional.directional.intensity, light.directional.intensity);
	// Spot
	light = SceneIO::LoadLight(jsn[3]);
	EXPECT_EQ(LightSpot.spot.position, light.spot.position);
	EXPECT_EQ(LightSpot.spot.direction, light.spot.direction);
	EXPECT_EQ(LightSpot.spot.coverageAngle, light.spot.coverageAngle);
	EXPECT_EQ(LightSpot.spot.falloffAngle, light.spot.falloffAngle);
	EXPECT_EQ(LightSpot.spot.coverageAngle, light.spot.coverageAngle);
	EXPECT_EQ(LightSpot.spot.color, light.spot.color);
	EXPECT_EQ(LightSpot.spot.intensity, light.spot.intensity);
	// Rectangular
	light = SceneIO::LoadLight(jsn[4]);
	EXPECT_EQ(LightRectangular.rectangular.position, light.rectangular.position);
	EXPECT_EQ(LightRectangular.rectangular.edge0, light.rectangular.edge0);
	EXPECT_EQ(LightRectangular.rectangular.edge1, light.rectangular.edge1);
	EXPECT_EQ(LightRectangular.rectangular.red, light.rectangular.red);
	EXPECT_EQ(LightRectangular.rectangular.green, light.rectangular.green);
	EXPECT_EQ(LightRectangular.rectangular.blue, light.rectangular.blue);
	EXPECT_EQ(LightRectangular.rectangular.intensity, light.rectangular.intensity);
}

TEST(SceneIOCommon, Transform)
{
	TransformStruct t;
	nlohmann::json jsn = ReadTestFile()[SceneIO::TRANSFORM_BASE];
	// First one is external, it should throw file not found
	EXPECT_THROW(SceneIO::LoadTransform(jsn[0]), SceneException);
	// Matrix
	t = SceneIO::LoadTransform(jsn[1]);
	EXPECT_EQ(Indentity4x4, t);
	// TRS
	t = SceneIO::LoadTransform(jsn[2]);
	EXPECT_EQ(Indentity4x4, t);
}