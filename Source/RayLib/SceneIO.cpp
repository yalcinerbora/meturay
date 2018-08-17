#include <fstream>
#include <json.hpp>

#include "SceneIO.h"
#include "Log.h"

// VS2017 lacking behind
#ifdef _WIN32
	#include <experimental\filesystem>
	namespace std { namespace filesystem = std::experimental::filesystem; }
#else
	#include <filesystem>
#endif

using namespace nlohmann;

static Vector3 JsonToVec3(const json& j)
{
	float x = j[0];
	float y = j[1];
	float z = j[2];
	return Vector3(x, y, z);
}

static json Vec3ToJson(const Vector3& v)
{
	json j = json::array();
	j.push_back(v[0]);
	j.push_back(v[1]);
	j.push_back(v[2]);
	return j;
}

static const char* VolumeTypeToString(const VolumeType& v)
{
	assert(static_cast<size_t>(v) < VolumeTypeSize);
	return SceneFile::VolumeTypes[static_cast<size_t>(v)];
}

static VolumeType StringToVolumeType(const std::string& s)
{
	for(size_t i = 0; i < VolumeTypeSize; i++)
	{
		if(std::string(SceneFile::VolumeTypes[i]) == s)
		{
			return static_cast<VolumeType>(i);
		}
	}
	assert(false);
	return VolumeType::END;
}

static json CameraToJson(const CameraPerspective& c)
{
	json r = json::object();
	r[SceneFile::TCameraGaze] = Vec3ToJson(c.gazePoint);
	r[SceneFile::TCameraPos] = Vec3ToJson(c.position);
	r[SceneFile::TCameraUp] = Vec3ToJson(c.up);
	r[SceneFile::TCameraAperture] = c.apertureSize;
	r[SceneFile::TCameraNear] = c.nearPlane;
	r[SceneFile::TCameraFar] = c.farPlane;
	r[SceneFile::TCameraFovX] = c.fov[0];
	r[SceneFile::TCameraFovY] = c.fov[1];
	return r;
}

static json VolumeToJson(const SceneFile::Volume& v)
{
	json r = json::object();
	r[SceneFile::TVolumeType] = VolumeTypeToString(v.type);
	r[SceneFile::TVolumeFile] = v.fileName;
	return r;
}

static json FMatToJson(const SceneFile::FluidMaterial& fMat)
{
	json r = json::object();
	//
	json colorArray = json::array(); 
	for(const auto& c : fMat.colors)
	{
		colorArray.push_back(Vec3ToJson(c));
	}
	r[SceneFile::TMaterialColors] = colorArray;
	//
	json colorFactors = json::array();
	for(const auto& c : fMat.colorInterp)
	{
		colorFactors.push_back(c);
	}
	r[SceneFile::TMaterialColorFactors] = colorFactors;
	//
	json opacities = json::array();
	for(const auto& o : fMat.opacities)
	{
		opacities.push_back(o);
	}
	r[SceneFile::TMaterialOpacities] = opacities;
	//
	json opacityFactors = json::array();
	for(const auto& o : fMat.opacityInterp)
	{
		opacityFactors.push_back(o);
	}
	r[SceneFile::TMaterialOpacityFactors] = opacityFactors;

	r[SceneFile::TMaterialId] = fMat.materialId;
	r[SceneFile::TMaterialAbsorbtionCoeff] = fMat.absorbtionCoeff;
	r[SceneFile::TMaterialScatteringCoeff] = fMat.scatteringCoeff;
	r[SceneFile::TMaterialTranparency] = Vec3ToJson(fMat.transparency);
	r[SceneFile::TMaterialFluidIOR] = fMat.ior;
	return r;
}

static SceneFile::FluidMaterial JsonToFMat(const json& jsonNode)
{
	SceneFile::FluidMaterial r;

	//
	//auto colorArray = ;
	for(const auto& c : jsonNode.at(SceneFile::TMaterialColors))
	{
		r.colors.push_back(JsonToVec3(c));
	}
	//
	auto colorFactors = jsonNode.at(SceneFile::TMaterialColorFactors);
	for(const float& c : colorFactors)
	{
		r.colorInterp.push_back(c);
	}
	//
	auto opacities = jsonNode.at(SceneFile::TMaterialOpacities);
	for(const float& o : opacities)
	{
		r.opacities.push_back(o);
	}
	//
	auto opacityFactors = jsonNode.at(SceneFile::TMaterialOpacityFactors);
	for(const float& o : opacityFactors)
	{
		r.opacityInterp.push_back(o);
	}	
	r.materialId = jsonNode[SceneFile::TMaterialId];
	r.absorbtionCoeff = jsonNode[SceneFile::TMaterialAbsorbtionCoeff];
	r.scatteringCoeff = jsonNode[SceneFile::TMaterialScatteringCoeff];
	r.transparency = JsonToVec3(jsonNode[SceneFile::TMaterialTranparency]);
	r.ior = jsonNode[SceneFile::TMaterialFluidIOR];
	return r;
}

void SceneFile::Clean()
{
	fileName.clear();
	cameras.clear();
	fluidMaterials.clear();
	volumes.clear();
}

IOError SceneFile::Load(SceneFile& s, const std::string& fileName)
{
	s.Clean();
	try
	{
		s.fileName = fileName;

		json jsonFile;
		std::ifstream fileIn(std::filesystem::u8path(fileName));
		if(!fileIn.is_open()) return IOError::FILE_NOT_FOUND;
		fileIn >> jsonFile;

		// Fast load
		// Volume
		auto volumes = jsonFile[TVolume][0];
		Volume v;
		v.fileName = volumes[TVolumeFile];
		v.type = StringToVolumeType(volumes[TVolumeType]);
		v.surfaceId = 0;
		v.materialId = 0;
		s.volumes.push_back(v);

		// Material
		auto material = jsonFile[TMaterialBatches][TMaterialFluid][0];
		FluidMaterial fm = JsonToFMat(material);
		s.fluidMaterials.push_back(fm);
	}
	catch(std::logic_error(e))
	{
		s.Clean();
		METU_ERROR_LOG("Scene(Json) Exception: %s.", e.what());
		return IOError::SCENE_CORRUPTED;
	}
	return IOError::OK;
}

IOError SceneFile::Save(const SceneFile& s,
						const std::string& fileName)
{
	// Camera
	json cameraArray = json::array();
	for(const CameraPerspective& c : s.cameras)
	{
		json camObject = CameraToJson(c);
		cameraArray.push_back(camObject);
	}

	// Volumes
	json volumeArray = json::array();
	for(const Volume& v : s.volumes)
	{
		json volObject = VolumeToJson(v);
		volumeArray.push_back(volObject);
	}

	// Materials
	json materialObject = json::object();
	// Fluid Materials
	json fluidMaterials = json::array();
	for(const FluidMaterial& fm : s.fluidMaterials)
	{
		fluidMaterials.push_back(FMatToJson(fm));
	}
	materialObject[TMaterialFluid] = fluidMaterials;
	// ....

	// Assignment of Objects
	json jsonFile;
	jsonFile[TCamera] = cameraArray;
	jsonFile[TMaterialBatches] = materialObject;
	jsonFile[TVolume] = volumeArray;
	
	// Actual writing
	std::ofstream outputFile(std::filesystem::u8path(fileName));
	if(!outputFile.is_open()) return IOError::FILE_NOT_FOUND;
	outputFile << jsonFile.dump(2);
	return IOError::OK;
}