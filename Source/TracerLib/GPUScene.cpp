#include "GPUScene.h"
#include "RayLib/SceneIO.h"

SceneError GPUScene::OpenFile(const std::string& fileName)
{
	sceneJson = ...
}

GPUScene::GPUScene(const std::string& s)
	: fileName(s)
	, currentTime(0.0)
{}

SceneError GPUScene::LoadScene(double time)
{
	OpenFile(fileName);

	SceneError e(SceneError::OK);
	if((e = LoadCommon(time)) != SceneError::OK) return e;
	if((e = LoadMaterials(time)) != SceneError::OK) return e;
	if((e = LoadSurfaces(time)) != SceneError::OK) return e;
	return e;
}

SceneError GPUScene::ChangeTime(double time)
{
	OpenFile(fileName);

	SceneError e(SceneError::OK);
	if((e = ChangeTimeCommon(time)) != SceneError::OK) return e;
	if((e = ChangeTimeMaterials(time)) != SceneError::OK) return e;
	if((e = ChangeTimeSurfaces(time)) != SceneError::OK) return e;
	return e;
}

////////////////////
// GPUSceneCommon //
////////////////////

SceneError GPUSceneCommon::LoadCommon(double time)
{
	sceneJson .
}

GPUSceneCommon::GPUSceneCommon(const std::string& s)
	: GPUScene(s)
{}

size_t GPUSceneCommon::UsedGPUMemory()
{
	return transformMemory.Size() + lightMemory.Size();
}

size_t GPUSceneCommon::UsedCPUMemory()
{
	return cameraMemory.size() * sizeof(CameraPerspective);
}

const LightStruct* GPUSceneCommon::LightsGPU()
{
	return dLights;
}

const TransformStruct* GPUSceneCommon::TransformsGPU()
{
	return dTransforms;
}

const CameraPerspective* GPUSceneCommon::CamerasCPU()
{
	return cameras;
}