#include <gtest/gtest.h>
#include <gmock/gmock.h>

// Common
#include "RayLib/System.h"
#include "RayLib/Log.h"
#include "RayLib/Constants.h"

// Tracer
#include "TracerLib/TracerLoader.h"
#include "TracerLib/GPUScene.h"
#include "TracerLib/TracerBase.h"

// Node
#include "RayLib/SelfNode.h"
#include "RayLib/TracerError.h"

// Visor Realted
#include "RayLib/VisorWindowInput.h"
#include "VisorGL/VisorGLEntry.h"

TEST(HelloTriangle, Test)
{
	EnableVTMode();

	TracerParameters tracerParams =
	{
		0,
	};

	// Load Tracer Genrator from DLL
	SharedLib testLib("Tracer-Test");
	LogicInterface tracerGenerator = TracerLoader::LoadTracerLogic(testLib,
																   "GenerateBasicTracer",
																   "DeleteBasicTracer");
	
	// Generate GPU List
	// We are a self node so only this GPU's ptrs
	std::vector<std::vector<CudaGPU>> gpuList(1);
	TracerError err = CudaSystem::Initialize(gpuList[0]);
	if(err != TracerError::OK)
		ASSERT_FALSE(true);

	// Load Scene
	GPUScene scene("TestScenes/helloTriangle.json",
				   gpuList, *tracerGenerator.get());
	SceneError scnE = scene.LoadScene(0.0);
	if(scnE != SceneError::OK)
		ASSERT_FALSE(true);

	// Finally generate logic after successfull load
	TracerBaseLogicI* logic;
	scnE = tracerGenerator.get()->GenerateBaseLogic(logic, tracerParams,
													scene.MaxMatIds(),
													scene.MaxAccelIds());
	if(scnE != SceneError::OK)
		ASSERT_FALSE(true);

	// Camera (Dont use scenes camera)
	//CameraPerspective cam = scene.CamerasCPU()[0];
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
	
	// Tracer Generation
	TracerBase tracerBase;
	// Get a Self-Node
	SelfNode selfNode;
	// Visor Input
	VisorWindowInput input(1.0, 1.0, 2.0, selfNode);
	// Window Params
	VisorOptions visorOpts;
	visorOpts.iFormat = pixFormat;
	visorOpts.iSize = {1280, 720};
	visorOpts.stereoOn = false;
	visorOpts.eventBufferSize = 128;

	// Window Loop
	auto visorView = CreateVisorGL(visorOpts);
	visorView->SetInputScheme(&input);

	// Init & Run tracer
	tracerBase.Initialize(*logic);
	tracerBase.GenerateCameraRays(cam, 1);
	while(tracerBase.Continue())
	{
		tracerBase.Render();
	}
	tracerBase.FinishSamples();

	// Main Poll Loop
	while(visorView->IsOpen())
	{
		// Before try to show do render loop



		visorView->Render();

		// Present Back Buffer
		visorView->ProcessInputs();
	}
}