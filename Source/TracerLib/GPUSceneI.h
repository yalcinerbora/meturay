#pragma once

#include <vector>

#include "RayLib/DeviceMemory.h"
#include "RayLib/Camera.h"
#include "RayLib/SceneStructs.h"

struct SceneError;

class GPUSceneI
{
	public:
		virtual							~GPUSceneI() = default;

		//
		virtual SceneError				LoadScene(const std::string&) = 0;
		virtual SceneError				ChangeTime(double) = 0;

		virtual size_t					UsedGPUMemory() = 0;
};

// Load Common Data of MRay Scene format.
class GPUSceneCommon : public GPUSceneI
{
	private:
		// GPU Memory
		DeviceMemory						transformMemory;
		DeviceMemory						lightMemory;
		std::vector<CameraPerspective>		cameraMemory;

	protected:
		// Access
		LightStruct*						dLights;
		TransformStruct*					dTransforms;
		CameraPerspective*					cameras;

		std::string							openedSceneName;
		double								currentTime;

	public:
		virtual								~GPUSceneCommon() = default;

		// Implementation
		SceneError							LoadScene(const std::string&) override;
		SceneError							ChangeTime(double) override;
		size_t								UsedGPUMemory() override;
};
