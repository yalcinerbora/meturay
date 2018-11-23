#pragma once

#include <vector>
#include <json.hpp>

#include "RayLib/Camera.h"
#include "RayLib/SceneStructs.h"

#include "DeviceMemory.h"
#include "AcceleratorDeviceFunctions.h"
#include "CudaConstants.h"

struct SceneError;
struct SceneFileNode;
class TracerLogicGeneratorI;

using NodeListing = std::set<SceneFileNode>;
using TypeNameNodeListings = std::map<std::string, NodeListing>;


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

		// CPU Helper Data		
		RequiredAccelBatches				requiredAccelGroupListings;
		RequiredMatBatches					requiredMatBatchListings;		
		std::map<uint32_t, BaseLeaf>		surfaceListings;

		// GPU Pointers
		LightStruct*						dLights;
		TransformStruct*					dTransforms;
		
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
																	 std::map<uint32_t, BaseLeaf>& surfaceListings,
																	 double time = 0.0);

		SceneError							GenerateMaterialGroups(TracerLogicGeneratorI*,
																   const TypeNameNodeListings&,
																   double time = 0.0);
		SceneError							GeneratePrimitiveGroups(TracerLogicGeneratorI*,
																	const TypeNameNodeListings&,
																	double time = 0.0);
				// Material Assignment
		SceneError							AssignMaterials(TracerLogicGeneratorI* l, double time,
															const RequestedMatBatches& requestedMatBatches,
															const MatBatchGPUPairings& requestedGPUIds,
															int boundaryMaterialGPUId);
		// Assign Accelerators using this material mapping
		SceneError							AssignAccelerators(TracerLogicGeneratorI*, double,
															   const RequestedAccelBatches& requestedAccelBatches,
															   const MaterialKeyListing& matHitKeyList);

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
		//
		SceneError							PartitionSceneData(TracerLogicGeneratorI* l, double time,
															   const std::vector<std::vector<CudaGPU>>&);
			

		// Access GPU
		const LightStruct*					LightsGPU();
		const TransformStruct*				TransformsGPU();
		// Access CPU
		const CameraPerspective*			CamerasCPU();

		// Further Required Data for Construction
		const SurfaceStruct*				SurfaceList;
};
