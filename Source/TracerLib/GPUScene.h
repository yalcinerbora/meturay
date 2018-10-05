#pragma once

#include <vector>

#include "RayLib/DeviceMemory.h"
#include "RayLib/Camera.h"
#include "RayLib/SceneStructs.h"

#include <json.hpp>

struct SceneError;
class TracerTypeGenerator;

class GPUScene
{
	private:
			SceneError					OpenFile(const std::string&);

	protected:
		virtual SceneError				LoadCommon(double time) = 0;
		virtual SceneError				LoadMaterials(double time) = 0;
		virtual SceneError				LoadSurfaces(double time) = 0;

		virtual SceneError				ChangeTimeCommon(double time) = 0;
		virtual SceneError				ChangeTimeMaterials(double time) = 0;
		virtual SceneError				ChangeTimeSurfaces(double time) = 0;

		// Current File & Time
		nlohmann::json					sceneJson;
		std::string						fileName;
		double							currentTime;

	public:
										GPUScene(const std::string&);
		virtual							~GPUScene() = default;

		//
		SceneError						LoadScene(double);
		SceneError						ChangeTime(double);

		virtual size_t					UsedGPUMemory() = 0;
		virtual size_t					UsedCPUMemory() = 0;
};

// Load Common Data of MRay Scene format.
class GPUSceneCommon : public GPUScene
{
	private:
		// GPU Memory
		DeviceMemory						transformMemory;
		DeviceMemory						lightMemory;

		// CPU Memory
		std::vector<CameraPerspective>		cameraMemory;
		//std::vector<AcceleraorStruct>		acceleratorDefinitionMemory;
		//std::vector<SurfaceStruct>			surfaceDefinitionMemory;
		
	protected:
		// Access GPU
		const LightStruct*					dLights;
		const TransformStruct*				dTransforms;
		// Access CPU
		const CameraPerspective*			cameras;
		//const AcceleraorStruct*			acceleratorDefinitions;
		//const SurfaceStruct*				surfaceDefinitions;

		// Implementations
		SceneError							LoadCommon(double time) override;

	public:
											GPUSceneCommon(const std::string&);
		virtual								~GPUSceneCommon() = default;

		// Implementation
		virtual size_t						UsedGPUMemory() override;
		virtual size_t						UsedCPUMemory() override;


		// Access GPU
		const LightStruct*					LightsGPU();
		const TransformStruct*				TransformsGPU();
		// Access CPU
		const CameraPerspective*			CamerasCPU();
};
