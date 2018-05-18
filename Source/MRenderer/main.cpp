

#include "RayLib/TracerThread.h"
#include "RayLib/System.h"
#include "RayLib/Log.h"

#include "TracerCUDA/TracerCUDAEntry.h"
#include "VisorGL/VisorGLEntry.h"

int main(int argc, const char* argv[])
{
	EnableVTMode();


	// First arg is scene name
	if(argc <= 1)
	{
		METU_ERROR_LOG("Insufficient args...");
		return 1;
	}

	// Create Cuda Tracer
	auto tracerI = CreateTracerCUDA();
	TracerThread tracer(*tracerI);

	// Set Scene
	tracer.ChangeScene(std::string(argv[1]));
	

	// Window Loop
	auto visorView = CreateVisorGL();
	visorView->ResetImageBuffer({1280, 720}, PixelFormat::RGB32_UNORM);
	// Main Poll Loop
	while(visorView->IsOpen())
	{
		// Do Stuff
		//...

		// Present Back Buffer
		visorView->Present();
	}
	return 0;
}