#include <gtest/gtest.h>
#include <gmock/gmock.h>

// Common
#include "RayLib/System.h"
#include "RayLib/Log.h"
#include "RayLib/Constants.h"

// Tracer
#include "TracerLib/TracerLoader.h"
#include "TracerLib/GPUScene.h"

// Visor Realted
#include "RayLib/VisorWindowInput.h"
#include "VisorGL/VisorGLEntry.h"

TEST(HelloTriangle, Test)
{
	EnableVTMode();

	// Load Tracer Genrator from DLL
	SharedLib testLib("Tracer-Test");
	LogicInterface tracerGenerator = TracerLoader::LoadTracerLogic(testLib,
																   "GenerateBasicTracer",
																   "DeleteBasicTracer");

	// Load Scene
	GPUScene scene("TestScenes/helloTriangle.json");	
	SceneError e = scene.LoadScene(tracerGenerator.get(), 0.0);
	if(e != SceneError::OK)
		ASSERT_FALSE(true);
	

	//...
	uint32_t seed = 0;



	// Camera
	float aspectRatio = 16.0f / 9.0f;
	CameraPerspective cam;
	cam.apertureSize = 1.0f;
	cam.farPlane = 100.0f;
	cam.nearPlane = 0.1f;
	cam.fov = Vector2f(MathConstants::DegToRadCoef * 70.0f,
					   MathConstants::DegToRadCoef * 70.0f * (1.0f / aspectRatio));
	cam.up = YAxis;
	cam.position = Vector3f(0.0f, 5.0f, -31.2f);
	cam.gazePoint = Vector3f(0.0f, 5.0f, 0.0f);

	// Start Tracer Thread and Set scene
	const PixelFormat pixFormat = PixelFormat::RGBA_FLOAT;

	//TracerThread tracer(*tracerI, *logic, seed);
	//tracer.ChangeScene(std::string(argv[1]));

	//tracer.ChangePixelFormat(pixFormat);
	//tracer.ChangeResolution(Vector2ui(1920, 1080));
	//tracer.ChangeSampleCount(5);
	//tracer.ChangeImageSegment(Vector2ui(0, 0), Vector2ui(1920, 1080));
	//tracer.ChangeParams(TracerParameters{10});
	//tracer.ChangeCamera(cam);
	//tracer.Start();

	// Visor Input
	//VisorWindowInput input(1.0, 1.0, 2.0, selfDistributor);

	VisorOptions opts;
	opts.iFormat = pixFormat;
	opts.iSize = {1280, 720};
	opts.stereoOn = false;
	opts.eventBufferSize = 128;

	// Window Loop
	auto visorView = CreateVisorGL(opts);
	//visorView->SetInputScheme(&input);

	// Main Poll Loop
	while(visorView->IsOpen())
	{
		visorView->Render();

		// Present Back Buffer
		visorView->ProcessInputs();
	}
//	tracer.Stop();
}