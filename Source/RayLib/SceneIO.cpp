#include "SceneIO.h"
#include <fstream>
#include "Log.h"

// VS2017 lacking behind
#ifdef _WIN32
	#include <experimental\filesystem>
	namespace std { namespace filesystem = std::experimental::filesystem; }
#else
	#include <filesystem>
#endif

using namespace nlohmann;

Vector3 SceneFile::JsonToVec3(const json& j)
{
	float x = j[0];
	float y = j[1];
	float z = j[2];
	return Vector3(x, y, z);
}

json SceneFile::Vec3ToJson(const Vector3& v)
{
	json j = json::array();
	j.push_back(v[0]);
	j.push_back(v[1]);
	j.push_back(v[2]);
	return j;
}

const char* SceneFile::VolumeTypeToString(const VolumeType& v)
{
	assert(static_cast<size_t>(v) < VolumeTypeSize);
	return VolumeTypes[static_cast<size_t>(v)];
}

VolumeType SceneFile::StringToVolumeType(const char* s)
{
	for(size_t i = 0; i < VolumeTypeSize; i++)
	{
		if(std::string(VolumeTypes[i]) == std::string(s))
		{
			return static_cast<VolumeType>(i);
		}
	}
	assert(false);
	return VolumeType::END;
}

json SceneFile::CameraToJson(const CameraPerspective& c)
{
	json r = json::object();
	r[TCameraGaze] = Vec3ToJson(c.gazePoint);
	r[TCameraPos] = Vec3ToJson(c.position);
	r[TCameraUp] = Vec3ToJson(c.up);
	r[TCameraAperture] = c.apertureSize;
	r[TCameraNear] = c.nearPlane;
	r[TCameraFar] = c.farPlane;
	r[TCameraFovX] = c.fov[0];
	r[TCameraFovY] = c.fov[1];
	return r;
}

json SceneFile::VolumeToJson(const Volume& v)
{
	json r = json::object();
	r[TVolumeType] = VolumeTypeToString(v.type);
	r[TVolumeFile] = v.fileName;
	return r;
}

json SceneFile::FMatToJson(const FluidMaterial& fMat)
{
	json r = json::object();
	r[TMaterialDiffuseAlbedo] = Vec3ToJson(fMat.diffuseAlbedo);
	r[TMaterialSpecularAlbedo] = Vec3ToJson(fMat.specularAlbedo);
	r[TMaterialFluidIOR] = fMat.ior;
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


		//...


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