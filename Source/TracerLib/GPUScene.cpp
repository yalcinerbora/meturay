#include "GPUScene.h"

#include "RayLib/SceneIO.h"
#include "RayLib/Types.h"
#include "RayLib/SceneFileNode.h"

#include "TracerLogicI.h"
#include "GPUAcceleratorI.h"
#include "GPUPrimitiveI.h"
#include "GPUMaterialI.h"
#include "TracerLogicGeneratorI.h"

#include <filesystem>
#include <set>

SceneError GPUScene::OpenFile(const std::string& fileName)
{
	// TODO: get a lightweight lexer and strip comments
	// from json since json does not support comments
	// now its only pure json iterating over a scene is
	// not convenient without comments.

	// Always assume filenames are UTF-8
	const auto path = std::filesystem::u8path(fileName);
	std::ifstream file(path);

	if(!file.is_open()) return SceneError::FILE_NOT_FOUND;
	// Parse Json
	file >> sceneJson;
	return SceneError::OK;
}

GPUScene::GPUScene(const std::string& s)
	: fileName(s)
	, currentTime(0.0)
{}

bool GPUScene::FindNode(nlohmann::json& jsn, const char* c)
{
	auto i = sceneJson.find(c);
	return(i != sceneJson.end());
};

SceneError GPUScene::GenIdLookup(std::map<uint32_t, uint32_t>& result,
								 const nlohmann::json& array, IdBasedNodeType t)
{
	result.clear();
	uint32_t i = 0;
	for(const auto& jsn : array)
	{
		auto r = result.emplace(jsn[SceneIO::ID], i);
		if(!r.second)
		{			
			unsigned int i = static_cast<int>(SceneError::DUPLICATE_ACCEL_ID) + t;
			return static_cast<SceneError::Type>(i);
		}
		i++;
	}
	return SceneError::OK;
}

void GPUScene::LoadCommon(double time)
{	
	// CPU Temp Data
	std::vector<LightStruct> lightsCPU;
	std::vector<TransformStruct> transformsCPU;

	// Lights
	nlohmann::json lightsJson;
	if(FindNode(lightsJson, SceneIO::LIGHT_BASE))
	{
		lightsCPU.resize(lightsJson.size());
		int i = 0;
		for(const auto& lightJson : lightsJson)
		{
			lightsCPU[i] = SceneIO::LoadLight(lightJson, time);
			i++;
		}
	};

	// Transforms
	nlohmann::json transformsJson;
	if(FindNode(transformsJson, SceneIO::TRANSFORM_BASE))
	{
		transformsCPU.resize(transformsJson.size());
		int i = 0;
		for(const auto& transformJson : transformsJson)
		{
			transformsCPU[i] = SceneIO::LoadTransform(transformJson, time);
			i++;
		}
	};

	// Allocate GPU and Load
	size_t transformSize = transformsCPU.size() * sizeof(TransformStruct);
	transformSize = AlignByteCount * ((transformSize + (AlignByteCount - 1)) / AlignByteCount);
	size_t lightSize = lightsCPU.size() * sizeof(LightStruct);
	
	memory = DeviceMemory(transformSize + lightSize);
	dTransforms = reinterpret_cast<TransformStruct*>(static_cast<Byte*>(memory));
	dLights = reinterpret_cast<LightStruct*>(static_cast<Byte*>(memory) + transformSize);
	
	CUDA_CHECK(cudaMemcpy(dLights, lightsCPU.data(), lightsCPU.size() * sizeof(LightStruct),
						  cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dTransforms, transformsCPU.data(), transformsCPU.size() * sizeof(TransformStruct),
						  cudaMemcpyHostToDevice));


	// Now Load Camera
	nlohmann::json camerasJson;
	if(FindNode(camerasJson, SceneIO::CAMERA_BASE))
	{
		cameraMemory.resize(camerasJson.size());
		int i = 0;
		for(const auto& cameraJson : camerasJson)
		{
			cameraMemory[i] = SceneIO::LoadCamera(cameraJson, time);
			i++;
		}
	};
}

SceneError GPUScene::LoadLogicRelated(TracerLogicGeneratorI* l, double time)
{
	using SurfaceId = uint32_t;
	SceneError e = SceneError::OK;
	std::set<SceneFileNode> emptySet;

	// Load Id Based Arrays
	// Surface Data
	// Primitive
	// Accelerator
	// Definitions
	nlohmann::json surfaces;
	nlohmann::json primitives;
	nlohmann::json surfaceData;
	nlohmann::json materials;
	nlohmann::json accelerators;
	if(FindNode(surfaces, SceneIO::SURFACE_BASE)) return SceneError::SURFACES_ARRAY_NOT_FOUND;
	if(FindNode(primitives, SceneIO::PRIMITIVE_BASE)) return SceneError::PRIMITIVES_ARRAY_NOT_FOUND;
	if(FindNode(surfaceData, SceneIO::SURFACE_DATA_BASE)) return SceneError::SURFACE_DATA_ARRAY_NOT_FOUND;
	if(FindNode(materials, SceneIO::MATERIAL_BASE)) return SceneError::MATERIALS_ARRAY_NOT_FOUND;
	if(FindNode(accelerators, SceneIO::ACCELERATOR_BASE)) return SceneError::ACCELERATORS_ARRAY_NOT_FOUND;
	
	std::map<uint32_t, uint32_t> primList;
	std::map<uint32_t, uint32_t> surfaceDataList;
	std::map<uint32_t, uint32_t> materialList;
	std::map<uint32_t, uint32_t> acceleratorList;
	GenIdLookup(primList, primitives, PRIMITIVE);
	GenIdLookup(surfaceDataList, surfaceData, SURFACE_DATA);
	GenIdLookup(materialList, materials, MATERIAL);
	GenIdLookup(acceleratorList, accelerators, ACCELERATOR);

	// Load Surface Nodes
	std::vector<SurfaceStruct> surfacesCPU;
	surfacesCPU.reserve(surfaceData.size());
	for(const auto& jsn : surfaces)
	{
		SurfaceStruct surfSturct = SceneIO::LoadSurface(jsn, time);
		surfacesCPU.push_back(surfSturct);
	}
	// Sort w.r.t. primitive type and accelerator type
	//std::sort(surfacesCPU.begin(), surfacesCPU.end());
	struct AccelGroupData
	{
		uint32_t				outerId;
		std::string				primName;
		SceneFileNode			accelNode;
		std::set<IdPairings>	matIdPairs;
	};
	struct MatBatchData
	{
		uint32_t			outerId;
		std::string			primType;
		std::string			matType;
		std::set<uint32_t>	matIds;
	};
	
	// Data groupings
	std::map<std::string, std::set<SceneFileNode>> matGroupNodes;
	std::map<std::string, std::set<SceneFileNode>> primGroupNodes;
	// Batch data
	std::map<std::string, AccelGroupData> accelGroupNodes;
	std::map<std::string, MatBatchData> matBatchNodes;
	// Key values for material and accelerator
	std::map<IdPairings, HitKey> matBatchMaterialListings;
	std::map<uint32_t, HitKey> accBatchAcceleratorListings;

	// Iterate over surfaces
	// and collect data for groups and batches
	for(const auto& surf : surfacesCPU)
	{
		std::string primGroupType;
		std::string accelGroupType;
		AccelGroupData* accelGroup;
		std::set<SceneFileNode>* primGroup;
		nlohmann::json accelNode;

		// Try Find Prim Group		
		if(auto loc = primList.find(surf.primitiveId);
		   loc == primList.end())
		{
			const auto jsnNode = primitives[loc->second];
			primGroupType = jsnNode[SceneIO::NAME];
			primGroup = &primGroupNodes.emplace(primGroupType, 
												emptySet).first->second;
		}		   
		else return SceneError::PRIMITIVE_ID_NOT_FOUND;

		// Now try find accelerator group
		// First find accelrator name
		if(auto loc = acceleratorList.find(surf.acceleratorId);
		   loc == acceleratorList.end())
		{
			accelNode = accelerators[loc->second];
			accelGroupType = accelNode[SceneIO::NAME];
		}
		else return SceneError::ACCEL_ID_NOT_FOUND;

		// Then combine name with primitive and find
		AccelGroupData accelGroupData = 
		{
			0,
			primGroupType,
			SceneFileNode{surf.acceleratorId, accelNode},
			std::set<IdPairings>()
		};
		accelGroup = &accelGroupNodes.emplace(accelGroupType + primGroupType,
											  std::make_pair(surf.primitiveId, 
															 accelGroupData)).first->second;

		// Start loading mats and surface datas
		// Iterate mat surface pairs
		for(int i = 0; i < surf.pairCount; i++)
		{
			const auto& pairs = surf.matDataPairs;
			const uint32_t surfId = pairs[i].second;
			const uint32_t matId = pairs[i].first;
							
			// Check if surfData[i] exits
			if(auto loc = surfaceDataList.find(surfId);
			   loc == surfaceDataList.end())
			{
				const auto jsnNode = surfaceData[loc->second];
				// Add surface to prim group
				primGroup->emplace(SceneFileNode{surfId, jsnNode});
			}
			else return SceneError::SURFACE_DATA_ID_NOT_FOUND;

			// Check if surfMats[i] exists
			if(auto loc = materialList.find(matId);
			   loc != materialList.end())
			{
				// Material exists
				// Append it to its group
				const auto jsnNode = materials[loc->second];
				std::string matTypeName = jsnNode[SceneIO::NAME];

				const auto& matGroup = matGroupNodes.emplace(matTypeName, 
															 emptySet).first;
				matGroup->second.emplace(SceneFileNode{matId, jsnNode});

				// Generate its mat batch also
				MatBatchData batchData = MatBatchData
				{
					0,
					primGroupType,
					matTypeName,
					std::set<uint32_t>()
				};
				const auto& matBatch = matBatchNodes.emplace(matTypeName + primGroupType,
															 batchData).first;
				matBatch->second.matIds.emplace(matId);
			}
			else return SceneError::MATERIAL_ID_NOT_FOUND;
		}
		// Looks like this pairing is valid
		// add it
		accelGroup->matIdPairs.emplace(surf.matDataPairs);
	}

	// After iteration we know 
	// which groups should be instatiated
	// 
	// Material Groups
	for(const auto& matGroup : matGroupNodes)
	{
		// Allocate mat group
		std::string matTypeName = matGroup.first;
		const auto& matNodes = matGroup.second;
		//
		GPUMaterialGroupI* matGroup;
		if(e = l->GetMaterialGroup(matGroup, matTypeName))
			return e;
		if(e = matGroup->InitializeGroup(matNodes, time))
			return e;
	}
	// Primitive Groups
	for(const auto& primGroup : primGroupNodes)
	{
		// Allocate mat group
		std::string primTypeName = primGroup.first;
		const auto& primNodes = primGroup.second;
		//
		GPUPrimitiveGroupI* primGroup;
		if(e = l->GetPrimitiveGroup(primGroup, primTypeName))
			return e;
		if(e = primGroup->InitializeGroup(primNodes, time))
			return e;
	}	
	// Material Batches
	for(auto& materialBatch : matBatchNodes)
	{
		const std::string& accelGroupName = materialBatch.first;
		const std::string& primTName = materialBatch.second.matType;
		const std::string& matTName = materialBatch.second.primType;
		
		GPUPrimitiveGroupI* pGroup;
		GPUMaterialGroupI* mGroup;
		l->GetPrimitiveGroup(pGroup, primTName);
		l->GetMaterialGroup(mGroup, matTName);

		GPUMaterialBatchI* matBatch;
		uint32_t id;
		if(e = l->GetMaterialBatch(matBatch, id, *mGroup, *pGroup))
			return e;



		// Generate Keys..
		const GPUMaterialGroupI& matGroup = *mGroup;
		for(const auto& matId : materialBatch.second.matIds)
		{
			uint32_t innerId = matGroup.InnerId(matId);
			HitKey key = HitKey::CombinedKey(id, innerId);
		}

		l->

	}

	// Re-iterate surfaces
	// Now populate material key listings and
	// Primitive key listings	
	std::map<SurfaceId, HitKey> surfaceMatKeys;
	for(auto& surf : surfacesCPU)
	{



		//GPUMaterialBatchI* matBatch;
		//if(e = l->GetMaterialBatch(matBatch, *mGroup, *pGroup))
		//	return e;
		//.......................................
	}

	//.....

	// Accelerator Groups & Batches
	for(const auto& accelGroup : accelGroupNodes)
	{
		const std::string& accelGroupName = accelGroup.first;
		const auto& primTName = accelGroup.second.primName;
		const auto& pairings = accelGroup.second.matIdPairs;
		const auto& accelNode = accelGroup.second.accelNode;

		GPUPrimitiveGroupI* pGroup;
		l->GetPrimitiveGroup(pGroup, primTName);

		GPUAcceleratorGroupI* aGroup;
		if(e = l->GetAcceleratorGroup(aGroup, *pGroup, dTransforms, accelGroupName))
			return e;
		if(e = aGroup->InitializeGroup(accelNode, materialKeyListings, pairings, time))
			return e;

		// Batch
		GPUAcceleratorBatchI* accelBatch;
		if(e = l->GetAcceleratorBatch(accelBatch, *accelGroup, *pGroup))
			return e;
	}
	// Re-iterate surfaces
	// Now populate material key listings and
	// Primitive key listings	
	std::map<SurfaceId, HitKey> acceleratorKeyListings;
	for(const auto& surf : surfacesCPU)
	{
		matBatchMaterialListings

		GPUMaterialBatchI* matBatch;
		if(e = l->GetMaterialBatch(matBatch, *mGroup, *pGroup))
			return e;
		.......................................
	}

	// Generate Base Logic

	// Generate Base Accelerator


	// All of the data is generated
	return SceneError::OK;
}

void GPUScene::ChangeCommon(double time)
{
	// TODO:
}

SceneError GPUScene::ChangeLogicRelated(TracerLogicGeneratorI*, double time)
{
	// TODO:
	return SceneError::OK;
}

size_t GPUScene::UsedGPUMemory()
{
	//return transformMemory.Size() + lightMemory.Size();
	return 0;
}

size_t GPUScene::UsedCPUMemory()
{
	//return cameraMemory.size() * sizeof(CameraPerspective);
	return 0;
}

SceneError GPUScene::LoadScene(TracerLogicGeneratorI* l, double time)
{
	SceneError e(SceneError::OK);
	try
	{
		OpenFile(fileName);
		LoadCommon(time);
		e = LoadLogicRelated(l, time);
	}
	catch (SceneException const& e)
	{
		return e;
	}
	catch(std::exception const&)
	{
		return SceneError::JSON_FILE_PARSE_ERROR;
	}
	return e;
}

SceneError GPUScene::ChangeTime(TracerLogicGeneratorI* l, double time)
{
	try
	{
		OpenFile(fileName);
		ChangeCommon(time);
		ChangeLogicRelated(l, time);
	}
	catch(SceneException const& e)
	{
		return e;
	}
	catch(std::exception const&)
	{
		return SceneError::JSON_FILE_PARSE_ERROR;
	}
	return SceneError::OK;
}

const LightStruct* GPUScene::LightsGPU()
{
	return dLights;
}

const TransformStruct* GPUScene::TransformsGPU()
{
	return dTransforms;
}

const CameraPerspective* GPUScene::CamerasCPU()
{
	return cameraMemory.data();
}