#include "SelfNode.h"
#include "Camera.h"
#include "TracerStructs.h"

SelfNode::SelfNode()
	: camFunc(nullptr)
	, timeFunc(nullptr)
	, optFunc(nullptr)
	, startStopFunc(nullptr)
	, pauseContFunc(nullptr)
	, sceneFunc(nullptr)

	, sendPerIteration(0)
	, imgStreamOn(false)	
	, currentFPS(30)
	, currentFrame(0)
{
	
}

void SelfNode::SendMaterialRays(const std::vector<MatBatchRayDataCPU> matRayData)
{}

void SelfNode::SendImage(const std::vector<Vector3f> image,
						 const Vector2ui resolution,
						 const Vector2ui offset,
						 const Vector2ui size)
{
	if(imgFunc) imgFunc(image, resolution, offset, size);
}

void SelfNode::SendError(uint32_t errorEnum, ErrorType e)
{
	// TODO::Error
}

bool SelfNode::ShouldSendImage(uint32_t renderCount)
{
	return imgStreamOn && (renderCount % sendPerIteration == 0);
}

void SelfNode::RequestSceneFile(const std::string& fileName)
{}

void SelfNode::AttachRecieveSceneFileFunc(FileRecieveFunc)
{}

void SelfNode::RequestSceneAccelerator()
{}

void SelfNode::RequestObjectAccelerator()
{}

void SelfNode::RequestObjectAccelerator(uint32_t objId)
{}

void SelfNode::RequestObjectAccelerator(const std::vector<uint32_t>& objIds)
{}

void SelfNode::AttachAcceleratorCallback(AcceleratorRecieveFunc)
{}

void SelfNode::WaitAccelerators()
{}

void SelfNode::WaitForMaterialRays(std::vector<MatBatchRayDataCPU>& matRayData)
{}

void SelfNode::AttachCameraCallback(SetCameraFunc f)
{
	camFunc = f;
}

void SelfNode::AttachTimeCallback(SetTimeFunc f)
{
	timeFunc = f;
}

void SelfNode::AttachOptionCallback(SetOptionFunc f)
{
	optFunc = f;
}

void SelfNode::AttachStartStopCallback(StartStopFunc f)
{
	startStopFunc = f;
}

void SelfNode::AttachPauseContCallback(PauseContFunc f)
{
	pauseContFunc = f;
}

void SelfNode::AttachSceneCallback(SetSceneFunc f)
{
	sceneFunc = f;
}

uint64_t SelfNode::TotalCPUMemory()
{
	return 0;
}

uint64_t SelfNode::TotalGPUMemory()
{
	return 0;
}

uint64_t SelfNode::GPUMemory(int GPUId)
{
	return 0;
}

bool SelfNode::Alone()
{
	return true;
}

void SelfNode::SetImageStream(bool b)
{
	imgStreamOn = b;
}

void SelfNode::SetImagePeriod(uint32_t iterations)
{
	sendPerIteration = iterations;
}

void SelfNode::ChangeCamera(const CameraPerspective& cam)
{
	if(camFunc) camFunc(cam);
}

void SelfNode::ChangeTime(double seconds)
{
	if(timeFunc) timeFunc(seconds);
}

void SelfNode::ChangeFPS(int fps)
{
	currentFPS = fps;
	if(timeFunc) timeFunc(1.0f / static_cast<double>(currentFPS) * currentFrame);
}

void SelfNode::NextFrame()
{
	currentFrame++;
	if(timeFunc) timeFunc(1.0f / static_cast<double>(currentFPS) * currentFrame);
}

void SelfNode::PreviousFrame()
{
	if(currentFrame != 0) currentFrame--;
	if(timeFunc) timeFunc(1.0f / static_cast<double>(currentFPS) * currentFrame);
}

void SelfNode::AttachDisplayCallback(SetImageSegmentFunc f)
{
	imgFunc = f;
}

void SelfNode::SetScene(const std::string& scene)
{
	if(sceneFunc) sceneFunc(scene);
}