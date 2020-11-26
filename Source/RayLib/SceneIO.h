#pragma once
/**

Scene file json interpeter and writer

*/
#include "SceneError.h"
#include "Vector.h"
#include "Matrix.h"
#include "Quaternion.h"
#include "SceneStructs.h"

#include <sstream>
#include <fstream>
#include <nlohmann/json.hpp>

struct CPUCamera;

template<class T>
using IntegerEnable = typename std::enable_if<std::is_integral<T>::value>::type;

template <class T>
struct Range
{
    // [start, end)
    T start;
    T end;
};

namespace SceneIO
{
    // Generic Time Dependency Check
    bool                    IsTimeDependent(const nlohmann::json&);

    // Load from anim
    template <class T>
    T                       LoadFromAnim(const std::string& fileName, double time = 0.0);

    // Static Loads
    bool                    LoadBool(const nlohmann::json&, double time = 0.0);
    std::string             LoadString(const nlohmann::json&, double time = 0.0);
    template <class T>
    T                       LoadNumber(const nlohmann::json&, double time = 0.0);
    template <class T>
    Quaternion<T>           LoadQuaternion(const nlohmann::json&, double time = 0.0);
    template <int N, class T>
    Vector<N, T>            LoadVector(const nlohmann::json&, double time = 0.0);
    template <int N, class T>
    Matrix<N, T>            LoadMatrix(const nlohmann::json&, double time = 0.0);

    // Ranged Arrays
    template <class T, typename = IntegerEnable<T>>
    std::vector<Range<T>>   LoadRangedNumbers(const nlohmann::json&);

    // Optional Fetch
    template <class T>
    T                       OptionalFetch(const nlohmann::json&, const char* name,
                                          T defaultValue);

    // Utility
    std::string             StripFileExt(const std::string& string);

    // Common Types
    uint32_t                LoadLightMatId(const nlohmann::json&);
    //LightType               LoadLightType(const nlohmann::json&);    
    //CPULight                LoadLight(const nlohmann::json&, double time = 0.0);

    //CPUCamera               LoadCamera(const nlohmann::json&, double time = 0.0);
    //CPUTransform            LoadTransform(const nlohmann::json&, double time = 0.0);
    //CPUMedium               LoadMedium(const nlohmann::json& jsn, double time = 0.0);
    //TextureStruct           LoadTexture(const nlohmann::json&, double time = 0.0);

    SurfaceStruct           LoadSurface(const nlohmann::json&);
    LightSurfaceStruct      LoadLightSurface(uint32_t baseMediumId,
                                             uint32_t identityTransformId,
                                             const nlohmann::json& jsn);
    CameraSurfaceStruct      LoadCameraSurface(uint32_t baseMediumId,
                                               uint32_t identityTransformId,
                                               const nlohmann::json& jsn);

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

inline bool SceneIO::LoadBool(const nlohmann::json& jsn, double time)
{
    if(IsTimeDependent(jsn))
        return LoadFromAnim<bool>(jsn, time);
    else if(jsn.is_boolean())
    {
        bool val = jsn;
        return val;
    }
    else throw SceneException(SceneError::TYPE_MISMATCH);
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

template <class T, typename>
std::vector<Range<T>> SceneIO::LoadRangedNumbers(const nlohmann::json& jsn)
{
    // It is mandatory for node to be an array    
    if(jsn.is_array())
    {
        std::vector<Range<T>> result;
        // Iterate
        for(const nlohmann::json& n : jsn)
        {
            // Each node can be either single integer or string
            // which can be decomposed of a range
            if(n.is_number())
            {
                T num = n;
                result.push_back(Range<T>{num, num + 1});
                
            }
            else if(n.is_string())
            {
                // Decompose range
                std::string range = n;
                std::stringstream s(range);
                T start, end;

                s >> start;
                s >> range;
                s >> end;
                result.push_back(Range<T>{start, end + 1});
            }
            else goto ERROR;
        }
        return result;
    }
    ERROR:
    throw SceneException(SceneError::TYPE_MISMATCH);
}

template <class T>
T SceneIO::OptionalFetch(const nlohmann::json& jsn, const char* name,
                         T defaultValue)
{
    auto iter = jsn.find(name);
    if(iter == jsn.end())
        return defaultValue;
    else
        return (*iter);
}

inline std::string SceneIO::StripFileExt(const std::string& string)
{
    return string.substr(string.find_last_of('.') + 1);
}