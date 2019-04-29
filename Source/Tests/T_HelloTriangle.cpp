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
#include "TracerLib/ScenePartitioner.h"

// Node
#include "RayLib/SelfNode.h"
#include "RayLib/TracerError.h"

// Visor Realted
#include "RayLib/VisorI.h"
#include "RayLib/VisorWindowInput.h"
#include "VisorGL/VisorGLEntry.h"

TEST(HelloTriangle, Test)
{
	EnableVTMode();

	static constexpr Vector2i IMAGE_RESOLUTION = {256, 256};
	static constexpr float ASPECT_RATIO = 1.0f;//16.0f / 9.0f;

	TracerParameters tracerParams =
	{
		0,
	};

	// Load Tracer Genrator from DLL
	SharedLib testLib("Tracer-Test");
	LogicInterface tracerGenerator = TracerLoader::LoadTracerLogic(testLib,
																   "GenerateBasicTracer",
																   "DeleteBasicTracer");

	// Generate GPU List & A Partitioner
	// Check cuda system error here
	
	const std::vector<CudaGPU>& gpuList = CudaSystem::GPUList();	
	if(CudaSystem::SystemStatus() != CudaSystem::OK)
		ASSERT_FALSE(true);
	int leaderDevice = gpuList[0].DeviceId();

	// GPU Data Partitioner
	SingleGPUScenePartitioner partitioner(gpuList);

	// Load Scene
	GPUScene scene("TestScenes/helloTriangle.json", partitioner,
				   *tracerGenerator.get());
	SceneError scnE = scene.LoadScene(0.0);
	if(scnE != SceneError::OK)
		ASSERT_FALSE(true);

	// Finally generate logic after successfull load
	TracerBaseLogicI* logic;
	scnE = tracerGenerator->GenerateBaseLogic(logic, tracerParams,
											  scene.MaxMatIds(),
											  scene.MaxAccelIds());
	if(scnE != SceneError::OK)
		ASSERT_FALSE(true);
	

	// Camera (Dont use scenes camera)
	//CameraPerspective cam = scene.CamerasCPU()[0];
	CameraPerspective cam;
	cam.apertureSize = 1.0f;
	cam.farPlane = 100.0f;
	cam.nearPlane = 0.1f;
	cam.fov = Vector2f(MathConstants::DegToRadCoef * 70.0f,
					   MathConstants::DegToRadCoef * 70.0f * (1.0f / ASPECT_RATIO));
	cam.up = YAxis;
	cam.position = Vector3f(0.0f, 5.0f, -31.2f);
	cam.gazePoint = Vector3f(0.0f, 5.0f, 0.0f);

	// Start Tracer Thread and Set scene
	const PixelFormat pixFormat = PixelFormat::RGBA_FLOAT;
	// Tracer Generation
	TracerBase tracerBase;	
	// Visor Input
	VisorWindowInput input(1.0, 1.0, 2.0);
	// Window Params
	VisorOptions visorOpts;
	visorOpts.iFormat = pixFormat;
	visorOpts.iSize = IMAGE_RESOLUTION;
	visorOpts.stereoOn = false;
	visorOpts.eventBufferSize = 128;
	
	// Create Visor
	auto visorView = CreateVisorGL(visorOpts);
	visorView->SetInputScheme(&input);

	// Attach the logic & Image format
	tracerBase.AttachLogic(*logic);
	tracerBase.SetImagePixelFormat(pixFormat);
	tracerBase.ResizeImage(IMAGE_RESOLUTION);
	tracerBase.ReportionImage();
	tracerBase.ResetImage();	

	// Tracer Init
	TracerError trcE = tracerBase.Initialize(leaderDevice);
	if(trcE != TracerError::OK)
		ASSERT_TRUE(false);

	// Get a Self-Node
	VisorI& v = *visorView;
	SelfNode selfNode(v, tracerBase);
	input.AttachVisorCallback(selfNode);
	tracerBase.AttachTracerCallbacks(selfNode);

	// Run tracer
	tracerBase.GenerateInitialRays(scene, 0, 1);
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