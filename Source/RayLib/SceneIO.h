#pragma once
/**

Scene file json interpeter and writer

*/
#include "IOError.h"
#include "Camera.h"
#include "Vector.h"

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
			std::vector<Vector3f>	colors;
			std::vector<float>		colorInterp;

			float					absorbtionCoeff;
			float					scatteringCoeff;

			std::vector<float>		opacities;
			std::vector<float>		opacityInterp;

			Vector3f				transparency;		
			float					ior;
		};

		//struct Volume : public Surface
		//{
		//	VolumeType			type;
		//	std::string			fileName;
		//};

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
		// Fluid Material
		static constexpr const char*		TMaterialFluid = "Fluid Materials";
		static constexpr const char*		TMaterialColors = "colors";
		static constexpr const char*		TMaterialColorFactors = "colorInterp";
		static constexpr const char*		TMaterialOpacities = "opacities";
		static constexpr const char*		TMaterialOpacityFactors = "opacityInterp";
		static constexpr const char*		TMaterialAbsorbtionCoeff = "absorbtion";
		static constexpr const char*		TMaterialScatteringCoeff = "scattering";
		static constexpr const char*		TMaterialTranparency = "transparency";
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
//		static constexpr const char*		VolumeTypes[VolumeTypeSize] = {"maya_ncache_fluid"};
	
	private:
		void								Clean();

	public:
		std::string							fileName;
		// Data Types
		std::vector<CameraPerspective>		cameras;
		// Materials
		std::vector<FluidMaterial>			fluidMaterials;
		// Volumes
//		std::vector<Volume>					volumes;
		// TODO: add full materials
		
		// 
		static IOError						Load(SceneFile&,
												 const std::string& fileName);
		static IOError						Save(const SceneFile&,
												 const std::string& fileName);
};