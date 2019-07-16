#include "SceneNodeJson.h"
#include "RayLib/SceneIO.h"
#include "RayLib/Log.h"

template <class T, LoadFunc<T> LoadF>
std::vector<T> SceneNodeJson::AccessSingle(const nlohmann::json& node,
                                           const std::string& name,
                                           double time) const
{
    const nlohmann::json& nodeInner = node[name];
    std::vector<T> result;
    result.reserve(indexIdPairs.size());

    for(const auto& list : indexIdPairs)
    {
        const InnerIndex i = list.second;
        const nlohmann::json& node = (indexIdPairs.size() > 1) ? nodeInner[i] : nodeInner;
        result.push_back(LoadF(nodeInner, time));
    }
    return std::move(result);
}

template <class T, LoadFunc<T> LoadF>
std::vector<T> SceneNodeJson::AccessList(const nlohmann::json& node,
                                         const std::string& name,
                                         double time) const
{
    const nlohmann::json& nodeInner = node[name];
    std::vector<T> result;
    result.reserve(indexIdPairs.size());

    for(const auto& list : indexIdPairs)
    {  
        const InnerIndex i = list.second;
        const nlohmann::json& node = (indexIdPairs.size() > 1) ? nodeInner[i] : nodeInner;

        for(const nlohmann::json& n : node)
            result.push_back(LoadF(n, time));
    }
    return std::move(result);
}

SceneNodeJson::SceneNodeJson(const nlohmann::json& jsn, NodeId id)
    : SceneNodeI(id)
    , node(jsn)
{}

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

std::string SceneNodeJson::Name() const
{
    return SceneIO::LoadString(node[SceneIO::NAME]);
}

std::string SceneNodeJson::Tag() const
{
    nlohmann::json::const_iterator i;
    if((i = node.find(SceneIO::TAG)) != node.end())
        return SceneIO::LoadString(node[SceneIO::TAG]);
    return "";
}

std::vector<std::string> SceneNodeJson::AccessString(const std::string& name, double time) const
{
    return AccessSingle<std::string, SceneIO::LoadString>(node, name, time);
}

std::vector<float> SceneNodeJson::AccessFloat(const std::string& name, double time) const
{
    return AccessSingle<float, SceneIO::LoadNumber<float>>(node, name, time);
}

std::vector<Vector2> SceneNodeJson::AccessVector2(const std::string& name, double time) const
{
    return AccessSingle<Vector2, SceneIO::LoadVector<2, float>>(node, name, time);
}

std::vector<Vector3> SceneNodeJson::AccessVector3(const std::string& name, double time) const
{
    return AccessSingle<Vector3, SceneIO::LoadVector<3, float>>(node, name, time);
}

std::vector<Vector4> SceneNodeJson::AccessVector4(const std::string& name, double time) const
{
    return AccessSingle<Vector4, SceneIO::LoadVector<4, float>>(node, name, time);
}

std::vector<Matrix4x4> SceneNodeJson::AccessMatrix4x4(const std::string& name, double time) const
{
    return AccessSingle<Matrix4x4, SceneIO::LoadMatrix<4, float>>(node, name, time);
}

std::vector<uint32_t> SceneNodeJson::AccessUInt(const std::string& name, double time) const
{
    return AccessSingle<uint32_t, SceneIO::LoadNumber<uint32_t>>(node, name, time);
}

std::vector<std::string> SceneNodeJson::AccessStringList(const std::string& name, double time) const
{
    return AccessList<std::string, SceneIO::LoadString>(node, name, time);
}

std::vector<float> SceneNodeJson::AccessFloatList(const std::string& name, double time) const
{
    return AccessList<float, SceneIO::LoadNumber<float>>(node, name, time);
}

std::vector<Vector2> SceneNodeJson::AccessVector2List(const std::string& name, double time) const
{
    return AccessList<Vector2, SceneIO::LoadVector<2, float>>(node, name, time);
}

std::vector<Vector3> SceneNodeJson::AccessVector3List(const std::string& name, double time) const
{
    return AccessList<Vector3, SceneIO::LoadVector<3, float>>(node, name, time);
}

std::vector<Vector4> SceneNodeJson::AccessVector4List(const std::string& name, double time) const
{
    return AccessList<Vector4, SceneIO::LoadVector<4, float>>(node, name, time);
}

std::vector<Matrix4x4> SceneNodeJson::AccessMatrix4x4List(const std::string& name, double time) const
{
    return AccessList<Matrix4x4, SceneIO::LoadMatrix<4, float>>(node, name, time);
}

std::vector<uint32_t> SceneNodeJson::AccessUIntList(const std::string& name, double time) const
{
    return AccessList<uint32_t, SceneIO::LoadNumber<uint32_t>>(node, name, time);
}