#include "SceneIO.h"
#include "Camera.h"
#include "Constants.h"

namespace SceneIO
{
	// Common Names
	static constexpr const char*	POSITION = "position";
	static constexpr const char*	DATA = "data";

	// Surface Related Names
	static constexpr const char*	TRANSFORM = "transform";
	static constexpr const char*	PRIMITIVE = "primitive";
	static constexpr const char*	ACCELERATOR = "accelerator";
	static constexpr const char*	MATERIAL = "material";
	
	// Camera Related Names	
	static constexpr const char*	CAMERA_APERTURE = "apertureSize";
	static constexpr const char*	CAMERA_FOCUS = "focusDistance";
	static constexpr const char*	CAMERA_PLANES = "planes";
	static constexpr const char*	CAMERA_FOV = "fov";
	static constexpr const char*	CAMERA_GAZE = "gaze";
	static constexpr const char*	CAMERA_UP = "up";

	// Light Related Names	
	// Light Type Values
	static constexpr const char*	LIGHT_PONT = "point";
	static constexpr const char*	LIGHT_DIRECTIONAL = "directional";
	static constexpr const char*	LIGHT_SPOT = "spot";
	static constexpr const char*	LIGHT_RECTANGULAR = "rectangular";
	// Common
	static constexpr const char*	LIGHT_COLOR = "color";
	static constexpr const char*	LIGHT_INTENSITY = "intensity";
	// Point
	// Directional
	static constexpr const char*	LIGHT_DIRECTION = "direction";
	// Spot
	static constexpr const char*	LIGHT_COVERAGE_ANGLE = "coverageAngle";
	static constexpr const char*	LIGHT_FALLOFF_ANGLE = "falloffAngle";
	// Rectangular
	static constexpr const char*	LIGHT_EDGE0 = "edge0";
	static constexpr const char*	LIGHT_EDGE1 = "edge1";
	// Transform Related Names			
	// Common
	static constexpr const char*	TRANSFORM_FORM = "form";
	// Transform Form Values
	static constexpr const char*	TRANSFORM_FORM_MATRIX4 = "matrix4x4";
	static constexpr const char*	TRANSFORM_FORM_T_R_S = "transformRotateScale";

	//
	LightStruct						LoadPoint(const nlohmann::json&, double time = 0.0);
	LightStruct						LoadDirectional(const nlohmann::json&, double time = 0.0);
	LightStruct						LoadSpot(const nlohmann::json&, double time = 0.0);
	LightStruct						LoadRectangular(const nlohmann::json&, double time = 0.0);

}

LightStruct SceneIO::LoadPoint(const nlohmann::json& jsn, double time)
{
	LightStruct s = {};
	s.t = LightType::POINT;
	s.point.position = LoadVector<3, float>(jsn[POSITION], time);
	s.point.color = LoadVector<3, float>(jsn[LIGHT_COLOR], time);
	s.point.intensity = LoadNumber<float>(jsn[LIGHT_INTENSITY], time);
	return s;
}

LightStruct SceneIO::LoadDirectional(const nlohmann::json& jsn, double time)
{
	LightStruct s = {};
	s.t = LightType::DIRECTIONAL;
	s.directional.direction = LoadVector<3, float>(jsn[LIGHT_DIRECTION], time);
	s.directional.color = LoadVector<3, float>(jsn[LIGHT_COLOR], time);
	s.directional.intensity = LoadNumber<float>(jsn[LIGHT_INTENSITY], time);
	return s;
}

LightStruct SceneIO::LoadSpot(const nlohmann::json& jsn, double time)
{
	LightStruct s = {};
	s.t = LightType::SPOT;
	s.spot.position = LoadVector<3, float>(jsn[POSITION], time);
	s.spot.direction = LoadVector<3, float>(jsn[LIGHT_DIRECTION], time);
	s.spot.intensity = LoadNumber<float>(jsn[LIGHT_INTENSITY], time);
	s.spot.falloffAngle = LoadNumber<float>(jsn[LIGHT_FALLOFF_ANGLE], time);
	s.spot.coverageAngle = LoadNumber<float>(jsn[LIGHT_COVERAGE_ANGLE], time);
	s.spot.color = LoadVector<3, float>(jsn[LIGHT_COLOR], time);
	return s;
}

LightStruct SceneIO::LoadRectangular(const nlohmann::json& jsn, double time)
{
	LightStruct s = {};
	s.t = LightType::SPOT;
	s.rectangular.position = LoadVector<3, float>(jsn[POSITION], time);		
	s.rectangular.edge0 = LoadVector<3, float>(jsn[LIGHT_EDGE0], time);
	s.rectangular.edge1 = LoadVector<3, float>(jsn[LIGHT_EDGE1], time);
	s.rectangular.intensity = LoadNumber<float>(jsn[LIGHT_INTENSITY], time);
	Vector3f color = LoadVector<3, float>(jsn[LIGHT_COLOR], time);
	s.rectangular.red = color[0];
	s.rectangular.green = color[1];
	s.rectangular.blue = color[2];
	return s;
}

TransformStruct SceneIO::LoadTransform(const nlohmann::json& jsn, double time)
{
	if(jsn.is_string())
	{
		return LoadFromAnim<TransformStruct>(jsn, time);
	}
	else if(jsn.is_object())
	{
		std::string type = LoadString(jsn[TYPE], time);
		if(type == TRANSFORM_FORM_MATRIX4)
		{
			Matrix4x4 mat = LoadMatrix<4, float>(jsn[DATA], time);
			return mat;
		}
		else if(type == TRANSFORM_FORM_T_R_S)
		{
			Vector3 translation = LoadVector<3, float>(jsn[DATA][0], time);
			Vector3 rotation = LoadVector<3, float>(jsn[DATA][1], time);
			Vector3 scale = LoadVector<3, float>(jsn[DATA][2], time);

			Matrix4x4 mat = TransformGen::Translate(translation);
			mat *= TransformGen::Scale(scale[0], scale[1], scale[2]);
			mat *= TransformGen::Rotate(rotation[2] * MathConstants::DegToRadCoef, ZAxis);
			mat *= TransformGen::Rotate(rotation[1] * MathConstants::DegToRadCoef, YAxis);
			mat *= TransformGen::Rotate(rotation[0] * MathConstants::DegToRadCoef, XAxis);
			
			return mat;
		}
		else throw SceneException(SceneError::UNKNOWN_TRANSFORM_TYPE);
		
	}
	else throw SceneException(SceneError::TYPE_MISMATCH);
}

CameraPerspective SceneIO::LoadCamera(const nlohmann::json& jsn, double time)
{
	if(jsn.is_string())
	{
		return LoadFromAnim<CameraPerspective>(jsn, time);
	}
	else if(jsn.is_object())
	{
		CameraPerspective cam = {};
		cam.position = LoadVector<3, float>(jsn[POSITION], time);
		cam.up = LoadVector<3, float>(jsn[CAMERA_UP], time);
		cam.gazePoint = LoadVector<3, float>(jsn[CAMERA_GAZE], time);
		Vector2 planes = LoadVector<2, float>(jsn[CAMERA_PLANES], time);
		cam.nearPlane = planes[0];
		cam.farPlane = planes[1];
		cam.fov = LoadVector<2, float>(jsn[CAMERA_FOV], time);
		cam.apertureSize = LoadNumber<float>(jsn[CAMERA_APERTURE], time);

		// Convert FOV to Radians
		cam.fov *= MathConstants::DegToRadCoef;

		return cam;
	}
	else throw SceneException(SceneError::TYPE_MISMATCH);
}

LightStruct SceneIO::LoadLight(const nlohmann::json& jsn, double time)
{
	if(jsn.is_string())
	{
		return LoadFromAnim<LightStruct>(jsn, time);
	}
	else if(jsn.is_object())
	{
		std::string type = jsn[TYPE];
		if(type == LIGHT_PONT)
		{
			return LoadPoint(jsn, time);
		}
		else if(type == LIGHT_DIRECTIONAL)
		{
			return LoadDirectional(jsn, time);
		}
		else if(type == LIGHT_SPOT)
		{
			return LoadSpot(jsn, time);
		}
		else if(type == LIGHT_RECTANGULAR)
		{
			return LoadRectangular(jsn, time);
		}
		else throw SceneException(SceneError::UNKNOWN_LIGHT_TYPE);
	}
	else throw SceneException(SceneError::TYPE_MISMATCH);
}

SurfaceStruct SceneIO::LoadSurface(const nlohmann::json& jsn, double time)
{
	if(jsn.is_string())
	{
		return LoadFromAnim<SurfaceStruct>(jsn, time);
	}
	else
	{
		SurfaceStruct s = {};
		s.transformId = jsn[TRANSFORM];
		s.acceleratorType = jsn[ACCELERATOR];
		s.matPrimPairs.fill(std::make_pair(std::numeric_limits<uint32_t>::max(),
										   std::numeric_limits<uint32_t>::max()));
		
		// Array Like Couples
		const auto primIdArray = jsn[PRIMITIVE];
		const auto materialIdArray = jsn[MATERIAL];
		if(primIdArray.size() != materialIdArray.size())
			throw SceneException(SceneError::PRIM_MATERIAL_NOT_SAME_SIZE);
		if(primIdArray.size() >= SceneConstants::MaxPrimitivePerSurface)
			throw SceneException(SceneError::TOO_MANY_SURFACE_ON_NODE);

		if(primIdArray.size() == 1)
		{
			s.matPrimPairs[0] = std::make_pair(materialIdArray, primIdArray);
		}
		else for(int i = 0; i < static_cast<int>(primIdArray.size()); i++)
			s.matPrimPairs[i] = std::make_pair(materialIdArray[i], primIdArray[i]);

		std::sort(s.matPrimPairs.begin(), s.matPrimPairs.end());
		s.pairCount = static_cast<int8_t>(primIdArray.size());
		return s;
	}
}