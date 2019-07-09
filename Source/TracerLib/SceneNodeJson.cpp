#include "SceneNodeJson.h"
#include "RayLib/SceneIO.h"

template <class T>
using LoadFunc = T(*)(const nlohmann::json&, double);

template <class T, LoadFunc<T>>
static std::vector<T> AccessList(nlohmann::json& node, const std::string& name, double time)
{
    nlohmann::json& nodeInner = node[name];
    std::vector<T> result;
    result.reserve(indexIdPairs.size());

    for(const auto& list : indexIdPairs)
    {
        const InnerIndex i = list.first;
        nlohmann::json& node = (nodeInner.is_array()) ? nodeInner[i] : nodeInner;

        result.push_back(LoadFunc(node, time));
    }
    return std::move(result);
}

SceneNodeJson::SceneNodeJson(nlohmann::json& jsn, NodeId id)
    : SceneNodeI(id)
    , node(jsn)
{}

// Interface
size_t SceneNodeJson::AccessCount() const
{
    return node.size();
}

size_t SceneNodeJson::AccessCount(const std::string& name) const
{
    return node[name].size();
}

std::string SceneNodeJson::AccessString(const std::string& name, double time) const
{
    return SceneIO::LoadString(node[name], time);
}

float SceneNodeJson::AccessFloat(const std::string& name, double time) const
{
    return SceneIO::LoadNumber<float>(node[name], time);
}

Vector2 SceneNodeJson::AccessVector2(const std::string& name, double time) const
{
    return SceneIO::LoadVector<2, float>(node[name], time);
}

Vector3 SceneNodeJson::AccessVector3(const std::string& name, double time) const
{
    return SceneIO::LoadVector<3, float>(node[name], time);
}

Vector4 SceneNodeJson::AccessVector4(const std::string& name, double time) const
{
    return SceneIO::LoadVector<4, float>(node[name], time);
}

Matrix4x4 SceneNodeJson::AccessMatrix4x4(const std::string& name, double time) const
{
    return SceneIO::LoadMatrix<4, float>(node[name], time);
}

uint32_t SceneNodeJson::AccessUInt(const std::string& name, double time) const
{
    return SceneIO::LoadNumber<uint32_t>(node[name], time);
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