#include <array>
#include <iostream>
#include <asio.hpp>

#include "RayLib/System.h"

#include "VisorGL/VisorGLEntry.h"

using asio::ip::tcp;


#include "RayLib/MayaCacheIO.h"
#include "RayLib/Log.h"

#include <fstream>

int main(int argc, const char* argv[])
{
	EnableVTMode();

	// Load Maya Cache
	MayaCache::MayaNSCacheInfo info;
	const std::string fileName("C:\\Users\\Coastal GPU\\Desktop\\CS568\\fluidCache.xml");
	if(IOError e; (e = MayaCache::LoadNCacheNavierStokesXML(info, fileName)) != IOError::OK)
	{
		METU_ERROR_LOG(GetIOErrorString(e));
		return 1;
	}

	// Load Cache
	std::vector<float> density, velocity;
	const std::string fileNameMCX("C:\\Users\\Coastal GPU\\Desktop\\CS568\\fluidCacheFrame1.mcx");
	if(IOError e; (e = MayaCache::LoadNCacheNavierStokes(density, velocity, info, fileNameMCX)) != IOError::OK)
	{
		METU_ERROR_LOG(GetIOErrorString(e));
		return 1;
	}
	
	// Load Successfull

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