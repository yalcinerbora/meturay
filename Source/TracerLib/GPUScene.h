#pragma once

#include <vector>
#include <nlohmann/json_fwd.hpp>

#include "RayLib/Camera.h"
#include "RayLib/SceneStructs.h"

#include "DeviceMemory.h"
#include "AcceleratorDeviceFunctions.h"
#include "CudaConstants.h"
#include "ScenePartitionerI.h"

struct SceneError;
struct SceneFileNode;
class ScenePartitionerI;
class TracerLogicGeneratorI;

using PrimitiveNodeList = std::map<std::string, NodeListing>;
using AcceleratorBatchList = std::map<std::string, AccelGroupData>;

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
		static constexpr const size_t			AlignByteCount = 128;

		// Fundamental
		TracerLogicGeneratorI&					logicGenerator;
		ScenePartitionerI&						partitioner;

		// Loaded
		Vector2i								maxAccelIds;
		Vector2i								maxMatIds;

		// GPU Memory
		DeviceMemory							memory;
		// CPU Memory
		std::vector<CameraPerspective>			cameraMemory;

		// File Related
		nlohmann::json*							sceneJson;
		std::string								fileName;
		double									currentTime;

		// GPU Pointers
		LightStruct*							dLights;
		TransformStruct*						dTransforms;

		// Inners
		// Helper Logic
		SceneError								OpenFile(const std::string& fileName);
		bool									FindNode(nlohmann::json& node, const char* name);
		static SceneError						GenIdLookup(std::map<uint32_t, uint32_t>&,
															const nlohmann::json& array,
															IdBasedNodeType);

		// Private Load Functionality
		SceneError		GenerateConstructionData(// Striped Listings (Striped from unsued nodes)
												 PrimitiveNodeList& primGroupNodes,
												 //
												 MaterialNodeList& matGroupNodes,
												 MaterialBatchList& matBatchListings,
												 AcceleratorBatchList& accelBatchListings,
												 // Base Accelerator required data
												 std::map<uint32_t, uint32_t>& surfaceTransformIds,
												 double time = 0.0);
		SceneError		GeneratePrimitiveGroups(const PrimitiveNodeList&,
												double time = 0.0);
		SceneError		GenerateMaterialGroups(const MultiGPUMatNodes&,
											   double time = 0.0);
		SceneError		GenerateMaterialBatches(MaterialKeyListing&,
												const MultiGPUMatBatches&,
												double time = 0.0);
		SceneError		GenerateAccelerators(std::map<uint32_t, AABB3>& accAABBs,
											 std::map<uint32_t, HitKey>& accHitKeyList,
											 //
											 const AcceleratorBatchList& acceleratorBatchList,
											 const MaterialKeyListing& matHitKeyList,
											 double time = 0.0);
		SceneError		GenerateBaseAccelerator(const std::map<uint32_t, AABB3>& accAABBs,
												const std::map<uint32_t, HitKey>& accHitKeyList,
												const std::map<uint32_t, uint32_t>& surfaceTransformIds,
												double time = 0.0);
		SceneError		GenerateBoundaryMaterial(int gpuId, double time = 0.0);

		void			LoadCommon(double time);
		SceneError		LoadLogicRelated(double time);

		void			ChangeCommon(double time);
		SceneError		ChangeLogicRelated(double time);

	public:
		// Constructors & Destructor
									GPUScene(const std::string&,
											 ScenePartitionerI& partitioner,
											 TracerLogicGeneratorI&);
									GPUScene(const GPUScene&) = delete;
									GPUScene(GPUScene&&);
		GPUScene&					operator=(const GPUScene&) = delete;
		//GPUScene&					operator=(GPUScene&&);
									~GPUScene();

		// Members
		size_t						UsedGPUMemory();
		size_t						UsedCPUMemory();
		//
		SceneError					LoadScene(double);
		SceneError					ChangeTime(double);
		//
		Vector2i					MaxMatIds();
		Vector2i					MaxAccelIds();
		// Access GPU
		const LightStruct*			LightsGPU() const;
		const TransformStruct*		TransformsGPU() const;
		// Access CPU
		const CameraPerspective*	CamerasCPU() const;
};
