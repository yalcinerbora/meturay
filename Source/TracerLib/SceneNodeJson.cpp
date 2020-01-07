#include "SceneNodeJson.h"
#include "RayLib/SceneIO.h"
#include "RayLib/Log.h"
#include "RayLib/SceneNodeNames.h"

template <class T, LoadFunc<T> LoadF>
std::vector<T> SceneNodeJson::AccessSingle(const std::string& name,
                                           double time) const
{
    const nlohmann::json& nodeInner = node[name];
    std::vector<T> result;
    result.reserve(indexIdPairs.size());

    for(const auto& list : indexIdPairs)
    {
        const InnerIndex i = list.second;
        const nlohmann::json& node = (indexIdPairs.size() > 1) ? nodeInner[i] : nodeInner;
        result.push_back(LoadF(node, time));
    }
    return std::move(result);
}

template <class T, LoadFunc<T> LoadF>
std::vector<std::vector<T>> SceneNodeJson::AccessList(const std::string& name,
                                                      double time) const
{
    const nlohmann::json& nodeInner = node[name];
    std::vector<std::vector<T>> result;
    result.reserve(indexIdPairs.size());

    for(const auto& list : indexIdPairs)
    {  
        result.emplace_back();

        const InnerIndex i = list.second;
        const nlohmann::json& node = (indexIdPairs.size() > 1) ? nodeInner[i] : nodeInner;

        for(const nlohmann::json& n : node)
            result.back().push_back(LoadF(n, time));
    }
    return std::move(result);
}

template <class T, LoadFunc<T> LoadF>
std::vector<T> SceneNodeJson::CommonList(const std::string& name,
                                         double time) const
{
    std::vector<T> result;
    const nlohmann::json& nodeInner = node[name];
    for(const nlohmann::json& n : nodeInner)
        result.push_back(LoadF(n, time));
    return std::move(result);
}

SceneNodeJson::SceneNodeJson(const nlohmann::json& jsn, NodeId id, bool forceFetchAll)
    : SceneNodeI(id)
    , node(jsn)
{
    if(forceFetchAll)
    {
        const auto idList = jsn[NodeNames::ID];
        if(idList.is_array())
        {
            uint32_t i = 0;
            for(uint32_t id : idList)
                AddIdIndexPair(id, i);
        }
        else AddIdIndexPair(idList, 0);
    }
}

std::string SceneNodeJson::Name() const
{
    return SceneIO::LoadString(node[NodeNames::NAME]);
}

std::string SceneNodeJson::Tag() const
{
    nlohmann::json::const_iterator i;
    if((i = node.find(NodeNames::TAG)) != node.end())
        return SceneIO::LoadString(node[NodeNames::TAG]);
    return "";
}

size_t SceneNodeJson::CommonListSize(const std::string& name) const
{
    const nlohmann::json& nodeInner = node[name];
    return nodeInner.size();
}

std::string SceneNodeJson::CommonString(const std::string& name, double time) const
{
    return SceneIO::LoadString(node[name], time);
}

float SceneNodeJson::CommonFloat(const std::string& name, double time) const
{
    return SceneIO::LoadNumber<float>(node[name], time);
}

Vector2 SceneNodeJson::CommonVector2(const std::string& name, double time) const
{
    return SceneIO::LoadVector<2, float>(node[name], time);
}

Vector3 SceneNodeJson::CommonVector3(const std::string& name, double time) const
{
    return SceneIO::LoadVector<3, float>(node[name], time);
}

Vector4 SceneNodeJson::CommonVector4(const std::string& name, double time) const
{
    return SceneIO::LoadVector<4, float>(node[name], time);
}
Matrix4x4 SceneNodeJson::CommonMatrix4x4(const std::string& name, double time) const
{
    return SceneIO::LoadMatrix<4, float>(node[name], time);
}

uint32_t SceneNodeJson::CommonUInt(const std::string& name, double time) const
{
    return SceneIO::LoadNumber<uint32_t>(node[name], time);
}

uint64_t SceneNodeJson::CommonUInt64(const std::string& name, double time) const
{
    return SceneIO::LoadNumber<uint64_t>(node[name], time);
}

std::vector<std::string> SceneNodeJson::CommonStringList(const std::string& name, double time) const
{
    return CommonList<std::string, SceneIO::LoadString>(name, time);
}

std::vector<float> SceneNodeJson::CommonFloatList(const std::string& name, double time) const
{
    return CommonList<float, SceneIO::LoadNumber>(name, time);
}

std::vector<Vector2> SceneNodeJson::CommonVector2List(const std::string& name, double time) const
{
    return CommonList<Vector2, SceneIO::LoadVector<2, float>>(name, time);
}

std::vector<Vector3> SceneNodeJson::CommonVector3List(const std::string& name, double time) const
{
    return CommonList<Vector3, SceneIO::LoadVector<3, float>>(name, time);
}
std::vector<Vector4> SceneNodeJson::CommonVector4List(const std::string& name, double time) const
{
    return CommonList<Vector4, SceneIO::LoadVector<4, float>>(name, time);
}

std::vector<Matrix4x4> SceneNodeJson::CommonMatrix4x4List(const std::string& name, double time) const
{
    return CommonList<Matrix4x4, SceneIO::LoadMatrix<4, float>>(name, time);
}

std::vector<uint32_t> SceneNodeJson::CommonUIntList(const std::string& name, double time) const
{
    return CommonList<uint32_t, SceneIO::LoadNumber>(name, time);
}

std::vector<uint64_t> SceneNodeJson::CommonUInt64List(const std::string& name, double time) const
{
    return CommonList<uint64_t, SceneIO::LoadNumber>(name, time);
}

size_t SceneNodeJson::AccessListTotalCount(const std::string& name) const
{
    const auto r = AccessListCount(name);
    return std::accumulate(r.begin(), r.end(), 0ull);
}

std::vector<size_t> SceneNodeJson::AccessListCount(const std::string& name) const
{
    const nlohmann::json& nodeInner = node[name];
    std::vector<size_t> result;
    result.reserve(indexIdPairs.size());

    for(const auto& list : indexIdPairs)
    {
        const InnerIndex i = list.second;
        const nlohmann::json& node = (indexIdPairs.size() > 1) ? nodeInner[i] : nodeInner;

        result.push_back(node.size());
    }
    return std::move(result);
}


std::vector<std::string> SceneNodeJson::AccessString(const std::string& name, double time) const
{
    return AccessSingle<std::string, SceneIO::LoadString>(name, time);
}

std::vector<float> SceneNodeJson::AccessFloat(const std::string& name, double time) const
{
    return AccessSingle<float, SceneIO::LoadNumber<float>>(name, time);
}

std::vector<Vector2> SceneNodeJson::AccessVector2(const std::string& name, double time) const
{
    return AccessSingle<Vector2, SceneIO::LoadVector<2, float>>(name, time);
}

std::vector<Vector3> SceneNodeJson::AccessVector3(const std::string& name, double time) const
{
    return AccessSingle<Vector3, SceneIO::LoadVector<3, float>>(name, time);
}

std::vector<Vector4> SceneNodeJson::AccessVector4(const std::string& name, double time) const
{
    return AccessSingle<Vector4, SceneIO::LoadVector<4, float>>(name, time);
}

std::vector<Matrix4x4> SceneNodeJson::AccessMatrix4x4(const std::string& name, double time) const
{
    return AccessSingle<Matrix4x4, SceneIO::LoadMatrix<4, float>>(name, time);
}

std::vector<uint32_t> SceneNodeJson::AccessUInt(const std::string& name, double time) const
{
    return AccessSingle<uint32_t, SceneIO::LoadNumber<uint32_t>>(name, time);
}

std::vector<uint64_t> SceneNodeJson::AccessUInt64(const std::string& name, double time) const
{
    return AccessSingle<uint64_t, SceneIO::LoadNumber<uint64_t>>(name, time);
}

std::vector<StringList> SceneNodeJson::AccessStringList(const std::string& name, double time) const
{
    return AccessList<std::string, SceneIO::LoadString>(name, time);
}

std::vector<FloatList> SceneNodeJson::AccessFloatList(const std::string& name, double time) const
{
    return AccessList<float, SceneIO::LoadNumber<float>>(name, time);
}

std::vector<Vector2List> SceneNodeJson::AccessVector2List(const std::string& name, double time) const
{
    return AccessList<Vector2, SceneIO::LoadVector<2, float>>(name, time);
}

std::vector<Vector3List> SceneNodeJson::AccessVector3List(const std::string& name, double time) const
{
    return AccessList<Vector3, SceneIO::LoadVector<3, float>>(name, time);
}

std::vector<Vector4List> SceneNodeJson::AccessVector4List(const std::string& name, double time) const
{
    return AccessList<Vector4, SceneIO::LoadVector<4, float>>(name, time);
}

std::vector<Matrix4x4List> SceneNodeJson::AccessMatrix4x4List(const std::string& name, double time) const
{
    return AccessList<Matrix4x4, SceneIO::LoadMatrix<4, float>>(name, time);
}

std::vector<UIntList> SceneNodeJson::AccessUIntList(const std::string& name, double time) const
{
    return AccessList<uint32_t, SceneIO::LoadNumber<uint32_t>>(name, time);
}

std::vector<UInt64List> SceneNodeJson::AccessUInt64List(const std::string& name, double time) const
{
    return AccessList<uint64_t, SceneIO::LoadNumber<uint64_t>>(name, time);
}