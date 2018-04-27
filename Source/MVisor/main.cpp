#include <array>
#include <iostream>
#include <asio.hpp>

#include "RayLib/System.h"

#include "VisorGL/VisorGLEntry.h"

#include "TracerCUDA/TracerCUDAEntry.h"

#include "RayLib/TracerI.h"
#include "RayLib/MayaCacheIO.h"
#include "RayLib/Log.h"
#include "RayLib/Camera.h"
#include "RayLib/ImageIO.h"

#include <fstream>

using asio::ip::tcp;

int main(int argc, const char* argv[])
{
	EnableVTMode();

	// Load Maya Cache
	MayaCache::MayaNSCacheInfo info;
	const std::string fileName("C:\\Users\\Coastal GPU\\Desktop\\CS568\\fluidCache.xml");
	//const std::string fileName("C:\\Users\\Coastal GPU\\Documents\\maya\\projects\\default\\cache\\nCache\\fluid\\smallFluid\\smallFluidShape1.xml");
	if(IOError e; (e = MayaCache::LoadNCacheNavierStokesXML(info, fileName)) != IOError::OK)
	{
		METU_ERROR_LOG(GetIOErrorString(e));
		return 1;
	}

	// Load Cache
	std::vector<float> velocityDensity;
	const std::string fileNameMCX("C:\\Users\\Coastal GPU\\Desktop\\CS568\\fluidCacheFrame200.mcx");
	//const std::string fileNameMCX("C:\\Users\\Coastal GPU\\Documents\\maya\\projects\\default\\cache\\nCache\\fluid\\smallFluid\\smallFluidShape1Frame4.mcx");
	if(IOError e; (e = MayaCache::LoadNCacheNavierStokes(velocityDensity, info, fileNameMCX)) != IOError::OK)
	{
		METU_ERROR_LOG(GetIOErrorString(e));
		return 1;
	}

	// CPU Load Successfull
	auto tracer = CreateTracerCUDA();
	tracer->Initialize();

	Vector2ui resolution = Vector2ui(512, 512);

	// Generate Camera Rays
	CameraPerspective cam;
	cam.apertureSize = 1.0f;
	cam.fov = Vector2f(MathConstants::DegToRadCoef * 60.0f,
					   MathConstants::DegToRadCoef * 60.0f);
	//cam.position = Vector3f(-15.7f, 15.59f, -4.708f);
	//cam.gazePoint = Vector3f(0.0f, 6.0f, 0.0f);
	cam.position = Vector3f(0.0f, 5.0f, 20.0f);
	cam.gazePoint = Vector3f(0.0f, 5.0f, 0.0f);
	cam.farPlane = 1000.0f;
	cam.nearPlane = 0.1f;
	cam.up = -YAxis;

	tracer->CS568GenerateCameraRays(cam, resolution, 1);

	// Load Fluid
	tracer->LoadFluidToGPU(velocityDensity, info.dim);

	// Ray loop
	tracer->LaunchRays(Vector3(1.0f, 0.0f, 0.0f),
					   info.dim,
					   Vector3(-5.0f, 0.0f, -5.0f),
					   Vector3(10.0f, 10.0f, 10.0f));

	// Get Image
	auto image = tracer->GetImage(resolution);

	// Write Image
	ImageIO::System().WriteAsPNG(image, resolution, "test.png");



	// Visor Determination
	std::unique_ptr<VisorGL> visorView = CreateVisorGL();
	visorView->ResetImageBuffer({1280, 720}, PixelFormat::RGB32_UNORM);
	// Main Poll Loop
	while(visorView->IsOpen())
	{
		// Do Stuff
		//...
		
		// Present Back Buffer
		visorView->Present();
	}
	
	//try
	//{
	//	if(argc != 2)
	//	{
	//		std::cerr << "Usage: client <host>" << std::endl;
	//		return 1;
	//	}

	//	asio::io_service io_service;

	//	tcp::resolver resolver(io_service);
	//	tcp::resolver::query query(argv[1], "daytime");
	//	tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

	//	tcp::socket socket(io_service);
	//	asio::connect(socket, endpoint_iterator);

	//	for(;;)
	//	{
	//		std::array<char, 128> buf;
	//		asio::error_code error;
	//		//size_t len = socket.read_some(asio::buffer(buf), error);

	//		if(error == asio::error::eof)
	//			break; // Connection closed cleanly by peer.
	//		else if(error)
	//			throw asio::system_error(error); // Some other error.

	//		//std::cout.write(buf.data(), len);
	//	}
	//}
	//catch(std::exception& e)
	//{
	//	std::cerr << e.what() << std::endl;
	//}

	return 0;
}