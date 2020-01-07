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
#include <nlohmann/json.hpp>

struct CameraPerspective;

namespace SceneIO
{
    // Generic Time Dependency Check
    bool                IsTimeDependent(const nlohmann::json&);

    // Load from anim
    template <class T>
    T                   LoadFromAnim(const std::string& fileName, double time = 0.0);

    // Static Loads
    std::string         LoadString(const nlohmann::json&, double time = 0.0);
    template <class T>
    T                   LoadNumber(const nlohmann::json&, double time = 0.0);
    template <class T>
    Quaternion<T>       LoadQuaternion(const nlohmann::json&, double time = 0.0);
    template <int N, class T>
    Vector<N, T>        LoadVector(const nlohmann::json&, double time = 0.0);
    template <int N, class T>
    Matrix<N, T>        LoadMatrix(const nlohmann::json&, double time = 0.0);

    // Utility
    std::string         StripFileExt(const std::string& string);

    // Common Types
    LightStruct         LoadLight(const nlohmann::json&, double time = 0.0);
    TransformStruct     LoadTransform(const nlohmann::json&, double time = 0.0);
    CameraPerspective   LoadCamera(const nlohmann::json&, double time = 0.0);

    SurfaceStruct       LoadSurface(const nlohmann::json&, double time = 0.0);
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

inline std::string SceneIO::StripFileExt(const std::string& string)
{
    return string.substr(string.find_last_of('.') + 1);
}