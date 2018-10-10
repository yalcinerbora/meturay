#include "GPUScene.h"
#include "RayLib/SceneIO.h"
#include "RayLib/Types.h"

#include "TracerLogicI.h"
#include "GPUAcceleratorI.h"
#include "GPUPrimitiveI.h"
#include "GPUMaterialI.h"

#include <filesystem>

using SortedSurfaces = std::map<SurfaceStruct*, std::vector<nlohmann::json>, decltype(SurfaceStruct::SortComparePrimitive)>;
static_assert(std::is_same<decltype(SurfaceStruct::SortComparePrimitive),
						   decltype(SurfaceStruct::SortComparePrimAccel)>::value);
static_assert(std::is_same<decltype(SurfaceStruct::SortComparePrimAccel),
						   decltype(SurfaceStruct::SortComparePrimMaterial)>::value);

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
	sceneJson << file;
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
	dTransforms = reinterpret_cast<const TransformStruct*>(static_cast<Byte*>(memory));
	dLights = reinterpret_cast<const LightStruct*>(static_cast<Byte*>(memory) + transformSize);
	
	CUDA_CHECK(cudaMemcpy(const_cast<LightStruct*>(dLights), 
						  lightsCPU.data(), lightsCPU.size() * sizeof(LightStruct),
						  cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(const_cast<TransformStruct*>(dTransforms),
						  transformsCPU.data(), transformsCPU.size() * sizeof(TransformStruct),
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
		hCameras = cameraMemory.data();
	};
}

SceneError GPUScene::LoadLogicRelated(TracerLogicGeneratorI* l, double time)
{
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
		surfacesCPU.push_back(SceneIO::LoadSurface(jsn));
	
	// Partition surface nodes for generation etcs
	SortedSurfaces primBasedSurfaces(SurfaceStruct::SortComparePrimitive);
	SortedSurfaces primAccelBasedSurfaces(SurfaceStruct::SortComparePrimAccel);
	SortedSurfaces primMaterialBasedSurfaces(SurfaceStruct::SortComparePrimMaterial);
	for(const auto& surface : surfacesCPU)
	{		
		auto locPrim = primBasedSurfaces.emplace(&surface, std::vector<nlohmann::json>());
		auto locPrimAccel = primAccelBasedSurfaces.emplace(&surface, std::vector<nlohmann::json>());
		auto locPrimMat = primMaterialBasedSurfaces.emplace(&surface, std::vector<nlohmann::json>());

		std::vector<nlohmann::json>& primVec = locPrim.first->second;
		std::vector<nlohmann::json>& primAcelVec = locPrimAccel.first->second;
		std::vector<nlohmann::json>& primMatVec = locPrimMat.first->second;
		
		// Push Surface Data for Primitive Generation
		auto loc = surfaceDataList.find(surface.dataId);
		if(loc == primList.end()) return SceneError::SURFACE_DATA_ID_NOT_FOUND;
		else primVec.push_back(surfaces[loc->second]);
		
		// Accelerator

		// Material
	}



	//for()
	//{
	//}

		const SurfaceStruct& surface = surfacesCPU[i - 1];
		// Check splits
		if(surfacesCPU[i].primitiveId != surface.primitiveId)
		{
			// Find Split load to system
			auto loc = primList.find(surface.primitiveId);
			if(loc == primList.end()) return SceneError::PRIMITIVE_ID_NOT_FOUND;
			const std::string& surfacePrimitiveType = surfaces[loc->second][SceneIO::TYPE];

			GPUPrimitiveGroupI* primGroup;
			SceneError e = l->GetPrimitiveGroup(primGroup, surfacePrimitiveType);
			if(e != SceneError::OK) return e;

			// Actual Load
			primGroup->LoadSurfaces(std::vector<nlohmann::>surfacesCPU, Vector2ui(start, i),
									{surfaceDataList, surfaceData});

			// Mark the other start
			start = i;
		}
	}


	// Then sort w.r.t. primitive/accelerator
	std::sort(surfacesCPU.begin(), surfacesCPU.end(), SurfaceStruct::SortComparePrimAccel);


	// Finally we sort w.r.t. primitive/material
	std::sort(surfacesCPU.begin(), surfacesCPU.end(), SurfaceStruct::SortComparePrimMaterial);


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
}

size_t GPUScene::UsedGPUMemory()
{
	//return transformMemory.Size() + lightMemory.Size();
}

size_t GPUScene::UsedCPUMemory()
{
	//return cameraMemory.size() * sizeof(CameraPerspective);
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
	catch(std::exception const& e)
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
	catch(std::exception const& e)
	{
		return SceneError::JSON_FILE_PARSE_ERROR;
	}
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
	return cameras;
}