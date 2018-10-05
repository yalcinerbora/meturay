#include "GPUSceneI.h"
#include "RayLib/SceneIO.h"

SceneError GPUSceneCommon::LoadScene(const std::string&)
{

	return SceneError::OK;
}

SceneError GPUSceneCommon::ChangeTime(double)
{
	return SceneError::OK;
}

size_t GPUSceneCommon::UsedGPUMemory()
{
	return transformMemory.Size() + lightMemory.Size();
}