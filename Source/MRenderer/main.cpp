#include "RayLib/System.h"
#include "RayLib/Log.h"
#include "RayLib/Constants.h"

#include "RayLib/SelfDistributor.h"
#include "RayLib/VisorWindowInput.h"

// DLLs
#include "TracerLib/TracerLoader.h"
#include "VisorGL/VisorGLEntry.h"

int main(int argc, const char* argv[])
{
	EnableVTMode();

	uint32_t seed = 0;
	//TracerLogicI* logic = nullptr;

	// Self Distributor
	//SelfDistributor selfDistributor;
	// Create Cuda Tracer
	//auto tracerI = CreateTracerCUDA();

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
	return 0;
}