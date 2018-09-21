#include "SelfDistributor.h"
#include "Camera.h"
//#include "RayHitStructs.h"

//SelfDistributor::SelfDistributor()
//	: camFunc(nullptr)
//	, timeFunc(nullptr)
//	, paramFunc(nullptr)
//	, startStopFunc(nullptr)
//	, pauseContFunc(nullptr)
//	, sceneFunc(nullptr)
//
//	, sendPerIteration(0)
//	, imgStreamOn(false)	
//	, currentFPS(30)
//	, currentFrame(0)
//{}
//
//void SelfDistributor::SendMaterialRays(uint32_t materialId,
//									   const RayRecordCPU)
//{}
//
//void SelfDistributor::SendMaterialRays(const std::vector<ArrayPortion<uint32_t>> materialIds,
//									   const RayRecordCPU)
//{}
//
//void SelfDistributor::SendImage(const std::vector<Vector3f> image,
//								const Vector2ui resolution,
//								const Vector2ui offset,
//								const Vector2ui size)
//{
//	if(imgFunc) imgFunc(image, resolution, offset, size);
//}
//
//void SelfDistributor::SendError(uint32_t errorEnum, ErrorType e)
//{
//	// TODO::Error
//}
//
//bool SelfDistributor::ShouldSendImage(uint32_t renderCount)
//{
//	return imgStreamOn && (renderCount % sendPerIteration == 0);
//}
//
//void SelfDistributor::RequestSceneFile(const std::string& fileName)
//{}
//
//void SelfDistributor::AttachRecieveSceneFileFunc(FileRecieveFunc)
//{}
//
//void SelfDistributor::RequestSceneAccelerator()
//{}
//
//void SelfDistributor::RequestObjectAccelerator()
//{}
//
//void SelfDistributor::RequestObjectAccelerator(uint32_t objId)
//{}
//
//void SelfDistributor::RequestObjectAccelerator(const std::vector<uint32_t>& objIds)
//{}
//
//void SelfDistributor::AttachAcceleratorCallback(AcceleratorRecieveFunc)
//{}
//
//void SelfDistributor::WaitAccelerators()
//{}
//
//void SelfDistributor::WaitForMaterialRays(RayRecordCPU&)
//{}
//
//void SelfDistributor::AttachCameraCallback(SetCameraFunc f)
//{
//	camFunc = f;
//}
//
//void SelfDistributor::AttachTimeCallback(SetTimeFunc f)
//{
//	timeFunc = f;
//}
//
//void SelfDistributor::AttachParamCallback(SetParameterFunc f)
//{
//	paramFunc = f;
//}
//
//void SelfDistributor::AttachStartStopCallback(StartStopFunc f)
//{
//	startStopFunc = f;
//}
//
//void SelfDistributor::AttachPauseContCallback(PauseContFunc f)
//{
//	pauseContFunc = f;
//}
//
//void SelfDistributor::AttachSceneCallback(SetSceneFunc f)
//{
//	sceneFunc = f;
//}
//
//uint64_t SelfDistributor::TotalCPUMemory()
//{
//	return 0;
//}
//
//uint64_t SelfDistributor::TotalGPUMemory()
//{
//	return 0;
//}
//
//bool SelfDistributor::Alone()
//{
//	return true;
//}
//
//void SelfDistributor::SetImageStream(bool b)
//{
//	imgStreamOn = b;
//}
//
//void SelfDistributor::SetImagePeriod(uint32_t iterations)
//{
//	sendPerIteration = iterations;
//}
//
//void SelfDistributor::ChangeCamera(const CameraPerspective& cam)
//{
//	if(camFunc) camFunc(cam);
//}
//
//void SelfDistributor::ChangeTime(double seconds)
//{
//	if(timeFunc) timeFunc(seconds);
//}
//
//void SelfDistributor::ChangeFPS(int fps)
//{
//	currentFPS = fps;
//	if(timeFunc) timeFunc(1.0f / static_cast<double>(currentFPS) * currentFrame);
//}
//
//void SelfDistributor::NextFrame()
//{
//	currentFrame++;
//	if(timeFunc) timeFunc(1.0f / static_cast<double>(currentFPS) * currentFrame);
//}
//
//void SelfDistributor::PreviousFrame()
//{
//	if(currentFrame != 0) currentFrame--;
//	if(timeFunc) timeFunc(1.0f / static_cast<double>(currentFPS) * currentFrame);
//}
//
//void SelfDistributor::AttachDisplayCallback(SetImageSegmentFunc f)
//{
//	imgFunc = f;
//}
//
//void SelfDistributor::SetScene(const std::string& scene)
//{
//	if(sceneFunc) sceneFunc(scene);
//}