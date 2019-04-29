#include "GPUScene.h"

#include "RayLib/SceneIO.h"
#include "RayLib/Types.h"

#include "TracerLogicI.h"
#include "GPUAcceleratorI.h"
#include "GPUPrimitiveI.h"
#include "GPUMaterialI.h"
#include "TracerLogicGeneratorI.h"
#include "SceneFileNode.h"
#include "ScenePartitionerI.h"

#include <nlohmann/json.hpp>
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
	sceneJson = new nlohmann::json();
	file >> (*sceneJson);
	return SceneError::OK;
}

GPUScene::GPUScene(const std::string& fileName,
				   ScenePartitionerI& partitioner,
				   TracerLogicGeneratorI& lg)
	: logicGenerator(lg)
	, partitioner(partitioner)
	, maxAccelIds(Vector2i(-1))
	, maxMatIds(Vector2i(-1))
	, fileName(fileName)
	, currentTime(0.0)
	, dLights(nullptr)
	, dTransforms(nullptr)
	, sceneJson(nullptr)
{}

GPUScene::GPUScene(GPUScene&& other)
	: logicGenerator(other.logicGenerator)
	, partitioner(other.partitioner)
	, maxAccelIds(other.maxAccelIds)
	, maxMatIds(other.maxMatIds)
	, memory(std::move(other.memory))
	, cameraMemory(std::move(other.cameraMemory))
	, sceneJson(other.sceneJson)
	, fileName(other.fileName)
	, currentTime(other.currentTime)	
	, dLights(other.dLights)
	, dTransforms(other.dTransforms)
{}

//GPUScene& GPUScene::operator=(GPUScene&& other)
//{
//	assert(this != &other);
//	logicGenerator = std::move(other.logicGenerator);
//	partitioner = std::move(other.partitioner);
//	maxAccelIds = other.maxAccelIds;
//	maxMatIds = other.maxMatIds;
//	memory = std::move(other.memory);
//	cameraMemory = std::move(other.cameraMemory);
//	sceneJson = other.sceneJson;
//	fileName = other.fileName;
//	currentTime = other.currentTime;
//	dLights = other.dLights;
//	dTransforms = other.dTransforms;
//	return *this;
//}

GPUScene::~GPUScene()
{
	if(sceneJson) delete sceneJson;
}

bool GPUScene::FindNode(nlohmann::json& jsn, const char* c)
{
	auto i = sceneJson->find(c);
	jsn = *i;
	return (i != sceneJson->end());
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

SceneError GPUScene::GenerateConstructionData(// Striped Listings (Striped from unsued nodes)											  
											  PrimitiveNodeList& primGroupNodes,
											  //
											  MaterialNodeList& matGroupNodes,
											  MaterialBatchList& matBatchListings,
											  AcceleratorBatchList& requiredAccelListings,
											  // Base Accelerator required data
											  std::map<uint32_t, uint32_t>& surfaceTransformIds,
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
				primSet.emplace(jsnNode, primId);
			}
			else return SceneError::PRIMITIVE_ID_NOT_FOUND;

			if(auto loc = materialList.find(matId);
			   loc != materialList.end())
			{
				const auto jsnNode = materials[loc->second];
				matGroupType = jsnNode[SceneIO::TYPE];
				auto& matSet = matGroupNodes.emplace(matGroupType,
													 std::set<SceneFileNode>()).first->second;
				matSet.emplace(jsnNode, matId);

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
		const auto& result = requiredAccelListings.emplace(acceleratorroupType, accGData).first;
		result->second.matPrimIdPairs.emplace(surfId, surf.matPrimPairs);

		// Generate transform pair also
		surfaceTransformIds.emplace(surfId, surf.transformId);
		surfId++;
	}
	return e;
}

SceneError GPUScene::GenerateMaterialGroups(const MultiGPUMatNodes& matGroupNodes,
											double time)
{
	// Generate Partitioned Material Groups
	SceneError e = SceneError::OK;
	for(const auto& matGroup : matGroupNodes)
	{
		const std::string& matTypeName = matGroup.first.first;
		const int gpuId = matGroup.first.second;
		const auto& matNodes = matGroup.second;
		//
		GPUMaterialGroupI* matGroup = nullptr;
		if(e = logicGenerator.GenerateMaterialGroup(matGroup, matTypeName, gpuId))
			return e;
		if(e = matGroup->InitializeGroup(matNodes, time))
			return e;
	}
	return e;
}

SceneError GPUScene::GenerateMaterialBatches(MaterialKeyListing& allMatKeys,
											 const MultiGPUMatBatches& materialBatches,
											 double time)
{
	SceneError e = SceneError::OK;
	// First do materials
	uint32_t batchId = BoundaryBatchId;
	for(const auto& requiredMat : materialBatches)
	{		
		batchId++;
		if(batchId >= (1 << HitKey::BatchBits))
			return SceneError::TOO_MANY_MATERIAL_GROUPS;

		const int gpuId = requiredMat.first.second;
		const std::string& batchName = requiredMat.first.first;
		const std::string& matTName = requiredMat.second.matType;
		const std::string& primTName = requiredMat.second.primType;

		GPUPrimitiveGroupI* pGroup = nullptr;
		GPUMaterialGroupI* mGroup = nullptr;
		logicGenerator.GeneratePrimitiveGroup(pGroup, primTName);
		logicGenerator.GenerateMaterialGroup(mGroup, matTName, gpuId);

		// Generation
		GPUMaterialBatchI* matBatch = nullptr;
		if((e = logicGenerator.GenerateMaterialBatch(matBatch, 
													 *mGroup, 
													 *pGroup,
													 batchId)) != SceneError::OK)
			return e;

		// Generate Keys
		// Find inner ids of those materials
		// And combine a key
		const GPUMaterialGroupI& matGroup = *mGroup;
		for(const auto& matId : requiredMat.second.matIds)
		{
			uint32_t innerId = matGroup.InnerId(matId);
			HitKey key = HitKey::CombinedKey(batchId, innerId);
			allMatKeys.emplace(std::make_pair(primTName, matId), key);

			maxMatIds = Vector2i::Max(maxMatIds, Vector2i(batchId, innerId));
		}
		batchId++;
	}
	return e;
}

SceneError GPUScene::GeneratePrimitiveGroups(const PrimitiveNodeList& primGroupNodes,
											 double time)
{
	// Generate Primitive Groups
	SceneError e = SceneError::OK;
	for(const auto& primGroup : primGroupNodes)
	{
		std::string primTypeName = primGroup.first;
		const auto& primNodes = primGroup.second;
		//
		GPUPrimitiveGroupI* primGroup = nullptr;
		if(e = logicGenerator.GeneratePrimitiveGroup(primGroup, primTypeName))
			return e;
		if(e = primGroup->InitializeGroup(primNodes, time))
			return e;
	}
	return e;
}

SceneError GPUScene::GenerateAccelerators(std::map<uint32_t, AABB3>& accAABBs,
										  std::map<uint32_t, HitKey>& accHitKeyList,
										  //
										  const AcceleratorBatchList& acceleratorBatchList,
										  const MaterialKeyListing& matHitKeyList,
										  double time)
{
	SceneError e = SceneError::OK;
	uint32_t accelBatch = 0;
	// Accelerator Groups & Batches and surface hit keys
	for(const auto& accelGroupBatch : acceleratorBatchList)
	{
		// Too many accelerators
		accelBatch++;
		if(accelBatch >= (1 << HitKey::BatchBits))
			return SceneError::TOO_MANY_ACCELERATOR_GROUPS;
	
		const uint32_t accelId = accelBatch;
		const std::string& accelGroupName = accelGroupBatch.second.accelType;
		const auto& primTName = accelGroupBatch.second.primType;
		const auto& pairings = accelGroupBatch.second.matPrimIdPairs;

		// Fetch Primitive
		GPUPrimitiveGroupI* pGroup = nullptr;
		if((e = logicGenerator.GeneratePrimitiveGroup(pGroup, primTName)) != SceneError::OK)
			return e;

		// Group Generation
		GPUAcceleratorGroupI* aGroup = nullptr;
		if((e = logicGenerator.GenerateAcceleratorGroup(aGroup, *pGroup, dTransforms, accelGroupName)) != SceneError::OK)
			return e;
		if((e = aGroup->InitializeGroup(matHitKeyList, pairings, time)) != SceneError::OK)
			return e;

		// Batch Generation
		GPUAcceleratorBatchI* aBatch = nullptr;
		if((e = logicGenerator.GenerateAcceleratorBatch(aBatch, *aGroup, *pGroup,
														accelId)) != SceneError::OK)
			return e;

		// Now Keys
		// Generate Accelerator Keys...
		const GPUAcceleratorGroupI& accGroup = *aGroup;
		for(const auto& pairings : accelGroupBatch.second.matPrimIdPairs)
		{
			const uint32_t surfId = pairings.first;
			uint32_t innerId = accGroup.InnerId(surfId);
			HitKey key = HitKey::CombinedKey(accelId, innerId);
			accHitKeyList.emplace(surfId, key);

			maxAccelIds = Vector2i::Max(maxAccelIds, Vector2i(accelId, innerId));

			// Attach keys of accelerators
			accHitKeyList.emplace(surfId, key);
		}

		// Find AABBs of these surfaces
		// For base Accelerator generation
		std::map<uint32_t, AABB3> aabbs;
		for(const auto& pairs : pairings)
		{
			AABB3 combinedAABB = ZeroAABB3;
			const IdPairings& pList = pairs.second;
			// Merge aabbs of the surfaces
			for(const auto& p : pList)
			{
				if(p.first == std::numeric_limits<uint32_t>::max()) break;

				AABB3 aabb = accGroup.PrimitiveGroup().PrimitiveBatchAABB(p.first);
				combinedAABB = combinedAABB.Union(aabb);
			}
			accAABBs.emplace(pairs.first, std::move(combinedAABB));
		}
	}
	return e;
}

SceneError GPUScene::GenerateBaseAccelerator(const std::map<uint32_t, AABB3>& accAABBs,
											 const std::map<uint32_t, HitKey>& accHitKeyList,
											 const std::map<uint32_t, uint32_t>& surfaceTransformIds,
											 double time)
{
	SceneError e = SceneError::OK;
	// Generate Surface Listings
	std::map<uint32_t, BaseLeaf> surfaceListings;
	for(const auto& pairs : surfaceTransformIds)
	{
		const uint32_t id = pairs.first;
		const AABB3f& aabb = accAABBs.at(id);
		const HitKey& key = accHitKeyList.at(id);

		BaseLeaf leaf =
		{
			aabb.Min(),
			key,
			aabb.Max(),
			pairs.second
		};
		surfaceListings.emplace(pairs.first, leaf);
	}

	// Find Base Accelerator Type and generate
	nlohmann::json baseAccel;
	if(!FindNode(baseAccel, SceneIO::BASE_ACCELERATOR_BASE))
		return SceneError::BASE_ACCELERATOR_NODE_NOT_FOUND;
	const std::string baseAccelType = baseAccel;

	// Generate Base Accelerator..
	GPUBaseAcceleratorI* baseAccelerator = nullptr;
	if((e = logicGenerator.GenerateBaseAccelerator(baseAccelerator, baseAccelType)) != SceneError::OK)
		return e;
	if((e = baseAccelerator->Initialize(surfaceListings)) != SceneError::OK)
		return e;
	return e;
}

SceneError GPUScene::GenerateBoundaryMaterial(int gpuId, double time)
{
	SceneError e = SceneError::OK;
	NodeListing nodeList;

	nlohmann::json node;
	if(!FindNode(node, SceneIO::BASE_OUTSIDE_MATERIAL))
		return SceneError::OUTSIDE_MAT_NODE_NOT_FOUND;
	nodeList.emplace(node);
	  
	GPUMaterialGroupI* boundaryMat = nullptr;
	const std::string matTypeName = node[SceneIO::TYPE];
	if((e = logicGenerator.GenerateBoundaryMaterial(boundaryMat, 
													matTypeName,
													gpuId)) != SceneError::OK)
		return e;
	if((e = boundaryMat->InitializeGroup(nodeList, time)) != SceneError::OK)
		return e;
	return e;
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

SceneError GPUScene::LoadLogicRelated(double time)
{
	SceneError e = SceneError::OK;
	// Group Data
	PrimitiveNodeList primGroupNodes;
	//
	MaterialNodeList matGroupNodes;	
	MaterialBatchList matListings;
	AcceleratorBatchList accelListings;
	std::map<uint32_t, uint32_t> surfaceTransformIds;

	// Parse Json and find necessary nodes
	if((e = GenerateConstructionData(primGroupNodes,
									 matGroupNodes,
									 matListings,
									 accelListings,
									 surfaceTransformIds,
									 time)) != SceneError::OK)
		return e;

	// Partition Material Data to Multi GPU Material Data
	int boundaryMaterialGPUId;
	MultiGPUMatNodes multiGPUMatNodes;
	MultiGPUMatBatches multiGPUMatBatches;
	if((e = partitioner.PartitionMaterials(multiGPUMatNodes,
										   multiGPUMatBatches,
										   boundaryMaterialGPUId,
										   //
										   matGroupNodes,
										   matListings)))
		return e;

	// Using those constructs generate
	// Primitive Groups
	if((e = GeneratePrimitiveGroups(primGroupNodes, time)) != SceneError::OK)
		return e;

	// Material Groups
	if((e = GenerateMaterialGroups(multiGPUMatNodes, time)) != SceneError::OK)
		return e;

	// Material Batches
	MaterialKeyListing allMaterialKeys;
	if((e = GenerateMaterialBatches(allMaterialKeys, 
									multiGPUMatBatches, 
									time)) != SceneError::OK)
		return e;

	// Accelerators
	std::map<uint32_t, AABB3> accAABBs;
	std::map<uint32_t, HitKey> accHitKeyList;
	if((e = GenerateAccelerators(accAABBs, accHitKeyList, accelListings,
								 allMaterialKeys, time)) != SceneError::OK)
		return e;

	// Base Accelerator
	if((e = GenerateBaseAccelerator(accAABBs, accHitKeyList, 
									surfaceTransformIds, time)) != SceneError::OK)
		return e;
	
	// Finally Boundary Material
	if((e = GenerateBoundaryMaterial(boundaryMaterialGPUId)) != SceneError::OK)
	   return e;

	// MaxIds are generated but those are inclusive
	// Make them exclusve
	maxAccelIds += Vector2i(1);
	maxMatIds += Vector2i(1);

	// Everything is generated!

	// All of the data is generated
	return SceneError::OK;
}

void GPUScene::ChangeCommon(double time)
{
	// TODO:
}

SceneError GPUScene::ChangeLogicRelated(double time)
{
	// TODO:
	return SceneError::BASE_ACCELERATOR_NODE_NOT_FOUND;
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

SceneError GPUScene::LoadScene(double time)
{
	SceneError e = SceneError::OK;
	try
	{
		if((e = OpenFile(fileName)) != SceneError::OK)
		   return e;
		LoadCommon(time);
		e = LoadLogicRelated(time);
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

SceneError GPUScene::ChangeTime(double time)
{
	try
	{
		OpenFile(fileName);
		ChangeCommon(time);
		ChangeLogicRelated(time);
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

Vector2i GPUScene::MaxMatIds()
{
	return maxMatIds;
}

Vector2i GPUScene::MaxAccelIds()
{
	return maxAccelIds;
}

const LightStruct* GPUScene::LightsGPU() const
{
	return dLights;
}

const TransformStruct* GPUScene::TransformsGPU() const
{
	return dTransforms;
}

const CameraPerspective* GPUScene::CamerasCPU() const
{
	return cameraMemory.data();
}