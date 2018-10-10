#pragma once
/**

Scene file json interpeter and writer

*/
#include "SceneError.h"
#include "Vector.h"
#include "Matrix.h"
#include "Quaternion.h"
#include "SceneStructs.h"

#include <fstream>
#include <json.hpp>

struct CameraPerspective;

namespace SceneIO
{
	static constexpr const char* SCENE_EXT = "mscene";
	static constexpr const char* ANIM_EXT = "manim";
	// Common Base Arrays
	static constexpr const char* CAMERA_BASE = "Cameras";
	static constexpr const char* LIGHT_BASE = "Lights";
	static constexpr const char* TRANSFORM_BASE = "Transforms";
	static constexpr const char* PRIMITIVE_BASE = "Primitives";
	static constexpr const char* ACCELERATOR_BASE = "Accelerators";
	static constexpr const char* MATERIAL_BASE = "Accelerators";
	static constexpr const char* SURFACE_BASE = "Surfaces";
	static constexpr const char* SURFACE_DATA_BASE = "SurfaceData";
	// Common Names
	static constexpr const char* ID = "id";
	static constexpr const char* TYPE = "type";
	static constexpr const char* name = "name";

	// Generic Time Dependency Check
	bool				IsTimeDependent(const nlohmann::json&);

	// Load from anim
	template <class T>
	T					LoadFromAnim(const std::string& fileName, double time = 0.0);

	// Static Loads	
	std::string			LoadString(const nlohmann::json&, double time = 0.0);
	template <class T>
	T					LoadNumber(const nlohmann::json&, double time = 0.0);	
	template <class T>
	Quaternion<T>		LoadQuaternion(const nlohmann::json&, double time = 0.0);
	template <int N, class T>
	Vector<N, T>		LoadVector(const nlohmann::json&, double time = 0.0);
	template <int N, class T>
	Matrix<N, T>		LoadMatrix(const nlohmann::json&, double time = 0.0);

	// Common Types
	LightStruct			LoadLight(const nlohmann::json&, double time = 0.0);
	TransformStruct		LoadTransform(const nlohmann::json&, double time = 0.0);
	CameraPerspective	LoadCamera(const nlohmann::json&, double time = 0.0);
};

inline bool SceneIO::IsTimeDependent(const nlohmann::json& jsn)
{
	std::string s;
	if(jsn.is_string() && (s = jsn).at(0) == '_')
	{
		return true;
	}
	return false;
}

template <class T>
T SceneIO::LoadFromAnim(const std::string& fileName, double time)
{
	// Strip '_'
	std::string filePath = fileName.substr(1);
	// Open File
	std::ifstream file(filePath);
	if(!file.is_open()) throw SceneException(SceneError::ANIMATION_FILE_NOT_FOUND);

	// TODO:
	// Read etc....
	return T();
}

inline std::string SceneIO::LoadString(const nlohmann::json& jsn, double time)
{
	if(IsTimeDependent(jsn))
		return LoadFromAnim<std::string>(jsn, time);
	else if(jsn.is_string())
	{
		std::string val = jsn;
		return val;
	}
	else throw SceneException(SceneError::TYPE_MISMATCH);
}

template <class T>
T SceneIO::LoadNumber(const nlohmann::json& jsn, double time) 
{
	if(IsTimeDependent(jsn))
		return LoadFromAnim<T>(jsn, time);
	else if(jsn.is_number() && jsn.size() == 1)
	{
		T val = jsn;
		return val;
	}
	else throw SceneException(SceneError::TYPE_MISMATCH);
}

template <class T>
Quaternion<T> SceneIO::LoadQuaternion(const nlohmann::json& jsn, double time)
{
	if(IsTimeDependent(jsn)) 
		return LoadFromAnim<Quaternion<T>>(jsn, time);
	else if(jsn.is_array() && jsn.size() == 4)
	{
		std::array<T, 4> array = jsn;
		return Quaternion<T>(array.data());
	}
	else throw SceneException(SceneError::TYPE_MISMATCH);
}

template <int N, class T>
Vector<N, T> SceneIO::LoadVector(const nlohmann::json& jsn, double time)
{
	if(IsTimeDependent(jsn))
		return LoadFromAnim<Vector<N, T>>(jsn, time);
	else if(jsn.is_array() && jsn.size() == N)
	{
		std::array<T, N> array = jsn;
		return Vector<N, T>(array.data());
	}
	else throw SceneException(SceneError::TYPE_MISMATCH);
}

template <int N, class T>
Matrix<N, T> SceneIO::LoadMatrix(const nlohmann::json& jsn, double time)
{
	if(IsTimeDependent(jsn))
		return LoadFromAnim<Matrix<N, T>>(jsn, time);
	else if(jsn.is_array() && jsn.size() == N * N)
	{
		std::array<T, N * N> array = jsn;
		return Matrix<N, T>(array.data());
	}
	else throw SceneException(SceneError::TYPE_MISMATCH);
}



//namespace SceneIO
//{
//	// Typeless vectors are defaulted to float
//	const auto LoadVector2 = LoadVector<2, float>;
//	const auto LoadVector3 = LoadVector<3, float>;
//	const auto LoadVector4 = LoadVector<4, float>;
//	// Float Type
//	const auto LoadVector2f = LoadVector<2, float>;
//	const auto LoadVector3f = LoadVector<3, float>;
//	const auto LoadVector4f = LoadVector<4, float>;
//	// Double Type
//	const auto LoadVector2d = LoadVector<2, double>;
//	const auto LoadVector3d = LoadVector<3, double>;
//	const auto LoadVector4d = LoadVector<4, double>;
//	// Integer Type
//	const auto LoadVector2i = LoadVector<2, int>;
//	const auto LoadVector3i = LoadVector<3, int>;
//	const auto LoadVector4i = LoadVector<4, int>;
//	// Unsigned Integer Type
//	const auto LoadVector2ui = LoadVector<2, unsigned int>;
//	const auto LoadVector3ui = LoadVector<3, unsigned int>;
//	const auto LoadVector4ui = LoadVector<4, unsigned int>;
//	// Quaternion Type
//	const auto LoadQuatF = LoadQuaternion<float>;
//	const auto LoadQuatD = LoadQuaternion<double>;
//	// Typeless vectors are defaulted to float
//	const auto LoadMatrix2x2 = LoadMatrix<2, float>;
//	const auto LoadMatrix3x3 = LoadMatrix<3, float>;
//	const auto LoadMatrix4x4 = LoadMatrix<4, float>;
//	// Float Type
//	const auto LoadMatrix2x2f = LoadMatrix<2, float>;
//	const auto LoadMatrix3x3f = LoadMatrix<3, float>;
//	const auto LoadMatrix4x4f = LoadMatrix<4, float>;
//	// Double Type
//	const auto LoadMatrix2x2d = LoadMatrix<2, double>;
//	const auto LoadMatrix3x3d = LoadMatrix<3, double>;
//	const auto LoadMatrix4x4d = LoadMatrix<4, double>;
//	// Integer Type
//	const auto LoadMatrix2x2i = LoadMatrix<2, int>;
//	const auto LoadMatrix3x3i = LoadMatrix<3, int>;
//	const auto LoadMatrix4x4i = LoadMatrix<4, int>;
//	// Unsigned Integer Type
//	const auto LoadMatrix2x2ui = LoadMatrix<2, unsigned int>;
//	const auto LoadMatrix3x3ui = LoadMatrix<3, unsigned int>;
//	const auto LoadMatrix4x4ui = LoadMatrix<4, unsigned int>;
//}