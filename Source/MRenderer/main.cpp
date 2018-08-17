

#include "RayLib/TracerThread.h"
#include "RayLib/System.h"
#include "RayLib/Log.h"

#include "RayLib/SelfDistributor.h"
#include "RayLib/VisorWindowInput.h"

// DLLs
#include "TracerCUDA/TracerCUDAEntry.h"
#include "VisorGL/VisorGLEntry.h"

#include "RayLib/SceneIO.h"

int main(int argc, const char* argv[])
{
	EnableVTMode();

	//float aspectRatio = (16.0f / 9.0f);
	//Vector2ui resolution(1280, 720);
	//CameraPerspective cam;
	//cam.apertureSize = 1.0f;
	//cam.farPlane = 100.0f;
	//cam.nearPlane = 0.1f;
	//cam.fov = Vector2f(MathConstants::DegToRadCoef * 70.0f,
	//				   MathConstants::DegToRadCoef * 60.0f * (1.0f / aspectRatio));
	//cam.up = YAxis;
	//cam.position = Vector3f(0.0f, 5.0f, 10.0f);
	//cam.gazePoint = Vector3f(0.0f, 5.0f, 0.0f);
	//SceneFile::FluidMaterial fMat;
	//fMat.colors = 
	//{	
	//	Vector3f(1.0f, 1.0f, 1.0f),
	//	Vector3f(1.0f, 1.0f, 1.0f),
	//	Vector3f(0.1278f, 0.1611f, 0.272f),
	//};
	//fMat.colorInterp = 
	//{
	//	0.0f,
	//	0.5913f,
	//	1.0f
	//};
	//fMat.opacities =
	//{
	//	0.0f,
	//	0.06f,
	//	1.0f
	//};
	//fMat.opacityInterp =
	//{
	//	0.0f,
	//	0.313f,
	//	1.0f
	//};
	//fMat.transparency = Vector3f(0.1814f);
	//fMat.absorbtionCoeff = 0.3f;
	//fMat.scatteringCoeff = 0.7f;
	//fMat.ior = 1.34f;
	//fMat.materialId = 1;
	//
	//SceneFile::Volume volume;
	//volume.materialId = 1;
	//volume.surfaceId = 1;
	//volume.type = VolumeType::MAYA_NCACHE_FLUID;
	//volume.fileName = "C:/Users/Coastal GPU/Desktop/CS568/fluidCache.xml";
	////
	//SceneFile s;
	//s.cameras.push_back(cam);
	//s.fluidMaterials.push_back(fMat);
	//s.volumes.push_back(volume);
	//auto e = s.Save(s, "testScene.jsn");
	//if(e != IOError::OK)
	//{
	//	return 1;
	//}


	// First arg is scene name
	if(argc <= 1)
	{
		METU_ERROR_LOG("Insufficient args...");
		return 1;
	}

	// Self Distributor
	SelfDistributor selfDistributor;
	// Create Cuda Tracer
	auto tracerI = CreateTracerCUDA();

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
	TracerThread tracer(*tracerI);
	tracer.ChangeScene(std::string(argv[1]));
	tracer.ChangeResolution(Vector2ui(1920, 1080));
	tracer.ChangeSampleCount(5);
	tracer.ChangeImageSegment(Vector2ui(0, 0), Vector2ui(1920, 1080));
	tracer.ChangeParams(TracerParameters{10});
	tracer.ChangeCamera(cam);
	tracer.Start(selfDistributor);

	// Visor Input
	VisorWindowInput input(1.0, 1.0, 2.0, selfDistributor);

	// Window Loop
	auto visorView = CreateVisorGL();
	visorView->ResetImageBuffer({1280, 720}, PixelFormat::RGB32_UNORM);
	// Main Poll Loop
	while(visorView->IsOpen())
	{
		// Do Stuff
		//...


		// Present Back Buffer
		//visorView->Present();
	}
	tracer.Stop();
	return 0;
}