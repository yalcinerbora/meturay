#pragma once

#include <vector>
#include <json.hpp>

#include "RayLib/Camera.h"
#include "RayLib/SceneStructs.h"

#include "DeviceMemory.h"

struct SceneError;
struct SceneFileNode;
class TracerLogicGeneratorI;

using NodeListing = std::set<SceneFileNode>;
using TypeNameNodeListings = std::map<std::string, NodeListing>;

struct AccelGroupData
{
	std::string						primName;
	std::map<uint32_t, IdPairings>	matPrimIdPairs;
};
struct MatBatchData
{
	std::string			primType;
	std::string			matType;
	std::set<uint32_t>	matIds;
};

class GPUScene
{
	public:
		enum IdBasedNodeType
		{
			ACCELERATOR,
			MATERIAL,
			PRIMITIVE,
			TRANSFORM,
			SURFACE_DATA
		};

	private:
		static constexpr const size_t		AlignByteCount = 128;

		// GPU Memory
		DeviceMemory						memory;
		// CPU Memory
		std::vector<CameraPerspective>		cameraMemory;		
		// File Related
		nlohmann::json						sceneJson;
		std::string							fileName;
		double								currentTime;
		
		// GPU Pointers
		LightStruct*						dLights;
		TransformStruct*					dTransforms;
			
		////
		//std::map<uint32_t, HitKey>			materialIdKeyPairings;
		//std::map<uint32_t, HitKey>			surfaceIdKeyPairings;

		// Inners
		// Helper Logic
		SceneError							OpenFile(const std::string& fileName);

		bool								FindNode(nlohmann::json& node, const char* name);
		static SceneError					GenIdLookup(std::map<uint32_t, uint32_t>&,
														const nlohmann::json& array,
														IdBasedNodeType);
		//
		SceneError							GenerateConstructionData(TracerLogicGeneratorI*,
																	 // Group Data
																	 std::map<std::string, std::set<SceneFileNode>>& matGroupNodes,
																	 std::map<std::string, std::set<SceneFileNode>>& primGroupNodes,
																	 // Batch Data
																	 std::map<std::string, AccelGroupData>& accelGroupListings,
																	 std::map<std::string, MatBatchData>& matBatchListings,
																	 // Base Accelerator required data
																	 std::map<uint32_t, uint32_t>& surfaceTransformListings,
																	 double time = 0.0);

		SceneError							GenerateMaterialGroups(TracerLogicGeneratorI*,
																   const TypeNameNodeListings&,
																   double time = 0.0);
		SceneError							GeneratePrimitiveGroups(TracerLogicGeneratorI*,
																	const TypeNameNodeListings&,
																	double time = 0.0);

		// Private Load Functionality
		void								LoadCommon(double time);
		SceneError							LoadLogicRelated(TracerLogicGeneratorI*, double time);

		void								ChangeCommon(double time);
		SceneError							ChangeLogicRelated(TracerLogicGeneratorI*, double time);


		//void								LoadMaterials(double time);
		//void								LoadSurfaces(double time);
		//
		//void								ChangeTimeCommon(double time);
		//void								ChangeTimeMaterials(double time);
		//void								ChangeTimeSurfaces(double time);
	public:
		// Constructors & Destructor
											GPUScene(const std::string&);
											GPUScene(const GPUScene&) = delete;
											GPUScene(GPUScene&&) = default;
		GPUScene&							operator=(const GPUScene&) = delete;
		GPUScene&							operator=(GPUScene&&) = default;
		virtual								~GPUScene() = default;

		// Members
		size_t								UsedGPUMemory();
		size_t								UsedCPUMemory();
		//
		SceneError							LoadScene(TracerLogicGeneratorI*, double);
		SceneError							ChangeTime(TracerLogicGeneratorI*, double);
		// Access GPU
		const LightStruct*					LightsGPU();
		const TransformStruct*				TransformsGPU();
		// Access CPU
		const CameraPerspective*			CamerasCPU();

		// Further Required Data for Construction
		const SurfaceStruct*				SurfaceList;
};
