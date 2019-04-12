#pragma once

#include <vector>
#include <nlohmann/json_fwd.hpp>

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
		static constexpr const size_t				AlignByteCount = 128;

		// Fundamental
		TracerLogicGeneratorI&						logicGenerator;
		const std::vector<CudaGPU>&					gpuList;

		// Loaded
		Vector2i									maxAccelIds;
		Vector2i									maxMatIds;

		// GPU Memory
		DeviceMemory								memory;
		// CPU Memory
		std::vector<CameraPerspective>				cameraMemory;		

		// File Related
		nlohmann::json*								sceneJson;
		std::string									fileName;
		double										currentTime;

		// CPU Helper Data		
		RequiredAccelBatches						requiredAccelGroupListings;
		RequiredMatBatches							requiredMatBatchListings;		
		std::map<uint32_t, BaseLeaf>				surfaceListings;

		// GPU Pointers
		LightStruct*								dLights;
		TransformStruct*							dTransforms;
		
		// Inners
		// Helper Logic
		SceneError							OpenFile(const std::string& fileName);
		bool								FindNode(nlohmann::json& node, const char* name);
		static SceneError					GenIdLookup(std::map<uint32_t, uint32_t>&,
														const nlohmann::json& array,
														IdBasedNodeType);

		// Private Load Functionality
		SceneError							GenerateConstructionData(// Group Data
																	 std::map<std::string, std::set<SceneFileNode>>& matGroupNodes,
																	 std::map<std::string, std::set<SceneFileNode>>& primGroupNodes,
																	 // Batch Data
																	 std::map<std::string, AccelGroupData>& accelGroupListings,
																	 std::map<std::string, MatBatchData>& matBatchListings,
																	 // Base Accelerator required data
																	 std::map<uint32_t, BaseLeaf>& surfaceListings,
																	 double time = 0.0);
		SceneError							GenerateMaterialGroups(const TypeNameNodeListings&,
																   double time = 0.0);
		SceneError							GeneratePrimitiveGroups(const TypeNameNodeListings&,
																	double time = 0.0);
		SceneError							PartitionSceneData(double);
		SceneError							AssignMaterials(double time,
															const RequestedMatBatches& requestedMatBatches,
															const MatBatchGPUPairings& requestedGPUIds,
															int boundaryMaterialGPUId);
		SceneError							AssignAccelerators(double time,
															   const RequestedAccelBatches& requestedAccelBatches,
															   const MaterialKeyListing& matHitKeyList);

		

		
		void								LoadCommon(double time);
		SceneError							LoadLogicRelated(double time);

		void								ChangeCommon(double time);
		SceneError							ChangeLogicRelated(double time);

	public:
		// Constructors & Destructor
											GPUScene(const std::string&,
													 const std::vector<CudaGPU>&,
													 TracerLogicGeneratorI&);
											GPUScene(const GPUScene&) = delete;
											GPUScene(GPUScene&&);
		GPUScene&							operator=(const GPUScene&) = delete;
		//GPUScene&							operator=(GPUScene&&);
											~GPUScene();

		// Members
		size_t								UsedGPUMemory();
		size_t								UsedCPUMemory();
		//
		SceneError							LoadScene(double);
		SceneError							ChangeTime(double);
		//
		Vector2i							MaxMatIds();
		Vector2i							MaxAccelIds();
		// Access GPU
		const LightStruct*					LightsGPU() const;
		const TransformStruct*				TransformsGPU() const;
		// Access CPU
		const CameraPerspective*			CamerasCPU() const;

		// Further Required Data for Construction
		const SurfaceStruct*				SurfaceList;
};
