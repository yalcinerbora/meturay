#pragma once
/**

Scene file json interpeter and writer

*/

#include "IOError.h"
#include <json.hpp>

#include "Camera.h"
#include "Vector.h"
#include "VolumeI.h"

struct SceneFile
{
	public:
		struct Material
		{
			uint32_t			materialId;
		};

		struct Surface
		{
			uint32_t			surfaceId;
			uint32_t			materialId;
		};

		struct FluidMaterial : public Material
		{
			Vector3				diffuseAlbedo;
			Vector3				specularAlbedo;
			float				ior;
		};

		struct Volume : public Surface
		{
			VolumeType			type;
			const std::string	fileName;
		};

	private:
		//  Camera Related
		static constexpr const char*		TCamera				= "Cameras";
		static constexpr const char*		TCameraGaze			= "gaze";
		static constexpr const char*		TCameraPos			= "pos";
		static constexpr const char*		TCameraNear			= "near";
		static constexpr const char*		TCameraFar			= "far";
		static constexpr const char*		TCameraUp			= "up";
		static constexpr const char*		TCameraAperture		= "aperture";
		static constexpr const char*		TCameraFovX			= "fovX";
		static constexpr const char*		TCameraFovY			= "fovY";
		
		// Light Related

		// Material Related
		static constexpr const char*		TMaterialBatches = "Materials";
		static constexpr const char*		TMaterialId = "material_id";
		static constexpr const char*		TMaterialDiffuseAlbedo = "d_albedo";
		static constexpr const char*		TMaterialSpecularAlbedo = "s_albedo";
		// Fluid Material
		static constexpr const char*		TMaterialFluid = "Fluid Materials";				
		static constexpr const char*		TMaterialFluidIOR = "ior";
				
		// Mesh Batch Related

		// Object & Volume Common
		static constexpr const char*		TSurfaceId = "surface_id";
		// Object Related
		// Volume Related
		static constexpr const char*		TVolume = "Volumes";
		static constexpr const char*		TVolumeType = "type";
		static constexpr const char*		TVolumeFile = "file";

		// Volume Type Strings
		static constexpr const char*		VolumeTypes[VolumeTypeSize] = {"maya_ncache_fluid"};
	
		
		void								Clean();
		// Conversion Utilities
		static Vector3						JsonToVec3(const nlohmann::json&);
		static nlohmann::json				Vec3ToJson(const Vector3&);

		static const char*					VolumeTypeToString(const VolumeType&);
		static VolumeType					StringToVolumeType(const char*);

		static nlohmann::json				CameraToJson(const CameraPerspective&);
		static nlohmann::json				VolumeToJson(const Volume&);
		static nlohmann::json				FMatToJson(const FluidMaterial&);

	public:
		std::string							fileName;
		// Data Types
		std::vector<CameraPerspective>		cameras;
		// Materials
		std::vector<FluidMaterial>			fluidMaterials;
		// Volumes
		std::vector<Volume>					volumes;
		// TODO: add full materials
		
		// 
		static IOError						Load(SceneFile&,
												 const std::string& fileName);
		static IOError						Save(const SceneFile&,
												 const std::string& fileName);
};