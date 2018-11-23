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
	, dLights(nullptr)
	, dTransforms(nullptr)
{}

bool GPUScene::FindNode(nlohmann::json& jsn, const char* c)
{
	auto i = sceneJson.find(c);
	jsn = *i;
	return (i != sceneJson.end());
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
			unsigned int i = static_cast<int>(SceneError::DUPLICATE_MATERIAL_ID) + t;
			return static_cast<SceneError::Type>(i);
		}
		i++;
	}
	return SceneError::OK;
}

SceneError GPUScene::GenerateConstructionData(TracerLogicGeneratorI*,
											  // Group Data
											  std::map<std::string, std::set<SceneFileNode>>& matGroupNodes,
											  std::map<std::string, std::set<SceneFileNode>>& primGroupNodes,
											  // Batch Data
											  std::map<std::string, AccelGroupData>& accelGroupListings,
											  std::map<std::string, MatBatchData>& matBatchListings,
											  // Base Accelerator required data
											  std::map<uint32_t, BaseLeaf>& surfaceListings,
											  double time)
{
	SceneError e = SceneError::OK;

	// Load Id Based Arrays
	nlohmann::json surfaces;
	nlohmann::json primitives;
	nlohmann::json materials;
	if(!FindNode(surfaces, SceneIO::SURFACE_BASE)) return SceneError::SURFACES_ARRAY_NOT_FOUND;
	if(!FindNode(primitives, SceneIO::PRIMITIVE_BASE)) return SceneError::PRIMITIVES_ARRAY_NOT_FOUND;
	if(!FindNode(materials, SceneIO::MATERIAL_BASE)) return SceneError::MATERIALS_ARRAY_NOT_FOUND;
	std::map<uint32_t, uint32_t> primList;
	std::map<uint32_t, uint32_t> materialList;
	GenIdLookup(primList, primitives, PRIMITIVE);
	GenIdLookup(materialList, materials, MATERIAL);

	// Iterate over surfaces
	// and collect data for groups and batches
	uint32_t surfId = 0;
	for(const auto& jsn : surfaces)
	{
		SurfaceStruct surf = SceneIO::LoadSurface(jsn, time);
		// Start loading mats and surface datas
		// Iterate mat surface pairs
		std::string primGroupType;
		for(int i = 0; i < surf.pairCount; i++)
		{
			const auto& pairs = surf.matPrimPairs;
			const uint32_t primId = pairs[i].second;
			const uint32_t matId = pairs[i].first;
			
			std::string matGroupType;
			// Check if primitive exists
			// add it to primitive group list for later construction
			if(auto loc = primList.find(primId);
			   loc != primList.end())
			{				
				const auto jsnNode = primitives[loc->second];
				std::string currentType = jsnNode[SceneIO::TYPE];
				if((i != 0) && primGroupType != currentType)
					return SceneError::PRIM_TYPE_NOT_CONSISTENT_ON_SURFACE;
				else primGroupType = currentType;
				auto& primSet = primGroupNodes.emplace(primGroupType,
													   std::set<SceneFileNode>()).first->second;
				primSet.emplace(SceneFileNode{primId, jsnNode});
			}
			else return SceneError::PRIMITIVE_ID_NOT_FOUND;

			if(auto loc = materialList.find(matId);
			   loc != materialList.end())
			{
				const auto jsnNode = materials[loc->second];
				matGroupType = jsnNode[SceneIO::TYPE];
				auto& matSet = matGroupNodes.emplace(matGroupType,
													   std::set<SceneFileNode>()).first->second;
				matSet.emplace(SceneFileNode{matId, jsnNode});

				// Generate its mat batch also
				MatBatchData batchData = MatBatchData
				{
					primGroupType,
					matGroupType,
					std::set<uint32_t>()
				};
				const auto& matBatch = matBatchListings.emplace(matGroupType + primGroupType,
																batchData).first;
				matBatch->second.matIds.emplace(matId);
			}
			else return SceneError::MATERIAL_ID_NOT_FOUND;
		}
		// Generate Accelerator Group
		const std::string acceleratorroupType = surf.acceleratorType + primGroupType;
		AccelGroupData accGData =
		{
			acceleratorroupType,
			primGroupType,
			std::map<uint32_t, IdPairings>()
		};
		const auto& result = accelGroupListings.emplace(acceleratorroupType, accGData).first;
		result->second.matPrimIdPairs.emplace(surfId, surf.matPrimPairs);

		// Generate transform pair also
		BaseLeaf leaf =
		{
			Vector3f(std::numeric_limits<float>::max()),
			HitKey::InvalidKey,
			Vector3f(-std::numeric_limits<float>::max()),
			surf.transformId
		};
		surfaceListings.emplace(surfId, leaf);
		surfId++;
	}
	return e;
}

SceneError GPUScene::GenerateMaterialGroups(TracerLogicGeneratorI* l,
											const TypeNameNodeListings& matGroupNodes, 
											double time)
{
	SceneError e = SceneError::OK;
	for(const auto& matGroup : matGroupNodes)
	{
		std::string matTypeName = matGroup.first;
		const auto& matNodes = matGroup.second;
		//
		GPUMaterialGroupI* matGroup;
		if(e = l->GetMaterialGroup(matGroup, matTypeName))
			return e;
		if(e = matGroup->InitializeGroup(matNodes, time))
			return e;
	}
	return e;
}

SceneError GPUScene::GeneratePrimitiveGroups(TracerLogicGeneratorI* l,
											 const TypeNameNodeListings& primGroupNodes,
											 double time)
{
	SceneError e = SceneError::OK;
	for(const auto& primGroup : primGroupNodes)
	{
		std::string primTypeName = primGroup.first;
		const auto& primNodes = primGroup.second;
		//
		GPUPrimitiveGroupI* primGroup;
		if(e = l->GetPrimitiveGroup(primGroup, primTypeName))
			return e;
		if(e = primGroup->InitializeGroup(primNodes, time))
			return e;
	}
	return e;
}

SceneError GPUScene::AssignMaterials(TracerLogicGeneratorI* l, double time,
									 const RequestedMatBatches& requestedMatBatches,
									 const MatBatchGPUPairings& requestedGPUIds,
									 int outsideMaterialGPUId)
{
	//
	SceneError e = SceneError::OK;

	auto matBatchIt = requestedMatBatches.begin();
	auto gpuIdIt = requestedGPUIds.begin();
	while(matBatchIt != requestedMatBatches.end())
	{
		const auto& materialBatch = *matBatchIt;
		const auto& gpuId = *gpuIdIt;

		//
		//const std::string& matGroupName = materialBatch.first;
		const std::string& matTName = materialBatch.second.matType;
		const std::string& primTName = materialBatch.second.primType;
		//
		GPUPrimitiveGroupI* pGroup = nullptr;
		GPUMaterialGroupI* mGroup = nullptr;
		l->GetPrimitiveGroup(pGroup, primTName);
		l->GetMaterialGroup(mGroup, matTName);
		// Generation
		GPUMaterialBatchI* matBatch;
		if((e = l->GenerateMaterialBatch(matBatch, *mGroup, *pGroup,
										 materialBatch.first,
										 gpuId.second)) != SceneError::OK)
			return e;

		// Allocation of Requested Materials on GPUs
		for(const uint32_t matId : materialBatch.second.matIds)
		{
			mGroup->LoadMaterial(matId, gpuId.second);
		}

		matBatchIt++;
		gpuIdIt++;
	}

	// Attach Outside Material
	auto outMatNode = sceneJson.find(SceneIO::BASE_OUTSIDE_MATERIAL);
	if(outMatNode == sceneJson.end()) return SceneError::OUTSIDE_MAT_NODE_NOT_FOUND;
	const nlohmann::json& node = *outMatNode;
	const std::string matTypeName = node[SceneIO::TYPE];

	// Generate its batch also
	GPUMaterialGroupI* matGroup;
	if((e = l->GetOutsideMaterial(matGroup, matTypeName, outsideMaterialGPUId)) != SceneError::OK)
		return e;

	// Initialize
	std::set<SceneFileNode> groupNodes = {SceneFileNode{node}};
	if((e = matGroup->InitializeGroup(groupNodes, time)) != SceneError::OK)
		return e;
	return SceneError::OK;
}

SceneError GPUScene::AssignAccelerators(TracerLogicGeneratorI* l, double time,
										const RequestedAccelBatches& requestedAccelBatches,
										const MaterialKeyListing& matHitKeyList)
{
	SceneError e = SceneError::OK;
	// Accelerator Groups & Batches
	// and surface hit keys
	std::map<uint32_t, HitKey> accHitKeyList;
	for(const auto& accelGroupBatch : requestedAccelBatches)
	{
		const std::string& accelGroupName = accelGroupBatch.second.accelType;
		const auto& primTName = accelGroupBatch.second.primType;
		const auto& pairings = accelGroupBatch.second.matPrimIdPairs;

		// Fetch Primitive
		GPUPrimitiveGroupI* pGroup;
		if((e = l->GetPrimitiveGroup(pGroup, primTName)) != SceneError::OK)
			return e;

		// Group Generation
		std::map<uint32_t, AABB3> aabbs;
		GPUAcceleratorGroupI* aGroup;
		if((e = l->GetAcceleratorGroup(aGroup, *pGroup, dTransforms, accelGroupName)) != SceneError::OK)
			return e;
		if((e = aGroup->InitializeGroup(aabbs, matHitKeyList, pairings, time)) != SceneError::OK)
			return e;

		// Attach aabbs to surface listings
		for(const auto& aabb : aabbs)
		{
			auto& baseLeaf = surfaceListings[aabb.first];
			baseLeaf.aabbMin = aabb.second.Min();
			baseLeaf.aabbMax = aabb.second.Max();
		}

		// Batch Generation
		GPUAcceleratorBatchI* accelBatch; //uint32_t id;
		if((e = l->GenerateAcceleratorBatch(accelBatch, *aGroup, *pGroup,
											accelGroupBatch.first)) != SceneError::OK)
			return e;

		// Now Keys
		// Generate Accelerator Keys...
		const GPUAcceleratorGroupI& accGroup = *aGroup;
		for(const auto& pairings : accelGroupBatch.second.matPrimIdPairs)
		{
			const uint32_t surfId = pairings.first;
			uint32_t innerId = accGroup.InnerId(surfId);
			HitKey key = HitKey::CombinedKey(accelGroupBatch.first, innerId);
			accHitKeyList.emplace(surfId, key);

			// Attach keys of accelerators
			surfaceListings[surfId].accKey = key;
		}
	}

	// Find Base Accelerator Type and generate
	nlohmann::json baseAccel;
	if(FindNode(baseAccel, SceneIO::BASE_ACCELERATOR_BASE))
		return SceneError::BASE_ACCELERATOR_NODE_NOT_FOUND;
	const std::string baseAccelType = baseAccel;

	// Generate Base Accelerator..
	GPUBaseAcceleratorI* baseAccelerator;
	if((e = l->GetBaseAccelerator(baseAccelerator, baseAccelType)) != SceneError::OK)
		return e;
	// Construct Base accelerator...
	baseAccelerator->Constrcut(surfaceListings);
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
	CUDA_CHECK(cudaMemcpy(dTransforms, transformsCPU.data(), 
						  transformsCPU.size() * sizeof(TransformStruct),
						  cudaMemcpyHostToDevice));
	if(lightsCPU.size() != 0)
	{
		dLights = reinterpret_cast<LightStruct*>(static_cast<Byte*>(memory) + transformSize);
		CUDA_CHECK(cudaMemcpy(dLights, lightsCPU.data(), lightsCPU.size() * sizeof(LightStruct),
							  cudaMemcpyHostToDevice));
	}
	
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
	SceneError e = SceneError::OK;
	// Group Data
	std::map<std::string, std::set<SceneFileNode>> matGroupNodes;
	std::map<std::string, std::set<SceneFileNode>> primGroupNodes;

	if((e = GenerateConstructionData(l,
									 matGroupNodes,
									 primGroupNodes,
									 requiredAccelGroupListings,
									 requiredMatBatchListings,
									 surfaceListings,
									 time)) != SceneError::OK)
		return e;

	// Using those constructs generate
	// Material Groups
	if((e = GenerateMaterialGroups(l, matGroupNodes, time)) != SceneError::OK)
		return e;
	// Primitive Groups
	if((e = GeneratePrimitiveGroups(l, primGroupNodes, time)) != SceneError::OK)
		return e;
	
	// Finally generate 
	TracerBaseLogicI* logic;
	if((e = l->GetBaseLogic(logic)) != SceneError::OK)
		return e;

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

SceneError GPUScene::PartitionSceneData(TracerLogicGeneratorI* l, double time, 
										const std::vector<std::vector<CudaGPU>>& nodeGPUList)
{
	SceneError e = SceneError::OK;

	MaterialKeyListing		allMatKeys;
	RequestedAccelBatches	accelBatches;
	RequestedMatBatches		materialBatches;
	MatBatchGPUPairings		gpuIdPairings;

	// TODO: Implement Multi-GPU Multi-PC partition code
	//=============================================================
	assert(nodeGPUList.size() == 1);
	assert(nodeGPUList[0].size() == 1);

	// Single GPU code
	int boundaryMatGPUId = 0;
	// First do materials
	uint32_t batchId = BoundaryBatchId;
	for(const auto& requiredMat : requiredMatBatchListings)
	{	
		batchId++;
		const std::string& accelGroupName = requiredMat.first;
		const std::string& matTName = requiredMat.second.matType;
		const std::string& primTName = requiredMat.second.primType;

		GPUPrimitiveGroupI* pGroup = nullptr;
		GPUMaterialGroupI* mGroup = nullptr;
		l->GetPrimitiveGroup(pGroup, primTName);
		l->GetMaterialGroup(mGroup, matTName);
		

		// Generate Requestd Batch as full batch of the material
		// Since single GPU, id is always 0
		materialBatches.emplace(batchId, requiredMat.second);	
		gpuIdPairings.emplace(batchId, 0);

		// Now Keys
		// Generate Keys...
		const GPUMaterialGroupI& matGroup = *mGroup;
		for(const auto& matId : requiredMat.second.matIds)
		{
			uint32_t innerId = matGroup.InnerId(matId);
			HitKey key = HitKey::CombinedKey(batchId, innerId);
			allMatKeys.emplace(std::make_pair(primTName, matId), key);
		}
	}
	// Then do accelerators
	batchId = 0;
	for(const auto& requiredAccel : requiredAccelGroupListings)
	{
		accelBatches.emplace(batchId, requiredAccel.second);
		batchId++;
	}
	// TODO: End
	// Here only your portion should be send to these two functions
	// below
	//=============================================================
	   
	// Do Loading
	if((e = AssignMaterials(l, time, materialBatches, gpuIdPairings,
							boundaryMatGPUId)) != SceneError::OK)
	{
		return e;
	}
	// Assign Accelerators using this material mapping
	if((e = AssignAccelerators(l, time, accelBatches, allMatKeys)) != SceneError::OK)
	{
		return e;
	}
	return e;
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