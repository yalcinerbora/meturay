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
    result.reserve(idIndexPairs.size());

    for(const auto& list : idIndexPairs)
    {
        const InnerIndex i = list.second;
        const nlohmann::json& node = (isMultiNode) ? nodeInner[i] : nodeInner;
        result.push_back(LoadF(node, time));
    }
    return std::move(result);
}

template <class T>
std::vector<T> SceneNodeJson::AccessRanged(const std::string& name) const
{
    const nlohmann::json& nodeInner = node[name];
    std::vector<T> result;
    result.reserve(idIndexPairs.size());

    if(!isMultiNode)
    {
        result.push_back(SceneIO::LoadNumber<T>(nodeInner));
    }
    else
    {
        std::vector<Range<T>> ranges = SceneIO::LoadRangedNumbers<T>(nodeInner);

        T total = 0;
        size_t j = 0;
        // Do single loop over pairings and ranges
        for(const auto& list : idIndexPairs)
        {
            // Find Data from the ranges
            const InnerIndex i = list.second;

            //for(const Range<T>& r : ranges)
            for(; j < ranges.size() ; j++)
            {
                Range<T> r = ranges[j];
                T count = r.end - r.start;
                if(i >= total && i < total + count)
                {
                    // Find the range now add the value
                    result.push_back(r.start + total - i);
                    break;
                }
                total += count;
            }
        }
    }
    return std::move(result);
}

template <class T, LoadFunc<T> LoadF>
std::vector<std::vector<T>> SceneNodeJson::AccessList(const std::string& name,
                                                      double time) const
{
    const nlohmann::json& nodeInner = node[name];
    std::vector<std::vector<T>> result;
    result.reserve(idIndexPairs.size());

    for(const auto& list : idIndexPairs)
    {
        result.emplace_back();

        const InnerIndex i = list.second;
        const nlohmann::json& node = (isMultiNode) ? nodeInner[i] : nodeInner;

        for(const nlohmann::json& n : node)
            result.back().push_back(LoadF(n, time));
    }
    return std::move(result);
}

template <class T, LoadFunc<T> LoadF>
OptionalNodeList<T> SceneNodeJson::AccessOptional(const std::string& name,
                                                  double time) const
{
    OptionalNodeList<T> result;
    result.reserve(idIndexPairs.size());

    // Check if entire data is missing then allocate for each innerId a empty node
    auto it = node.cend();
    if((it = node.find(name)) == node.cend())
    {
        OptionalNode<T> optNode;
        optNode.first = false;
        result.resize(idIndexPairs.size(), optNode);
        return std::move(result);
    }

    // Access the node and continue
    const nlohmann::json& nodeInner = node[name];

    for(const auto& list : idIndexPairs)
    {
        const InnerIndex i = list.second;
        const nlohmann::json& node = (isMultiNode) ? nodeInner[i] : nodeInner;

        OptionalNode<T> optNode;
        if(node.is_string())
        {
            std::string s = node;
            if(s == "")
                optNode.first = false;
        }
        else
        {
            optNode.first = true;
            optNode.second = LoadF(node, time);
        }
        result.push_back(optNode);
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

template <class T, LoadFunc<T> LoadF>
TexturedDataNodeList<T> SceneNodeJson::AccessTextured(const std::string& name,
                                                      double time) const
{
    const nlohmann::json& nodeInner = node[name];
    TexturedDataNodeList<T> result;
    result.reserve(idIndexPairs.size());

    for(const auto& list : idIndexPairs)
    {
        const InnerIndex i = list.second;
        const nlohmann::json& node = (isMultiNode) ? nodeInner[i] : nodeInner;

        TexturedDataNode<T> texNode;
        if(node.is_object())
        {
            NodeTextureStruct n = SceneIO::LoadNodeTextureStruct(node, time);
            texNode.isTexture = true;
            texNode.texNode = n;
        }
        else
        {
            texNode.isTexture = false;
            texNode.data = LoadF(node, time);
        }
        result.push_back(texNode);
    }
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
            isMultiNode = true;
            uint32_t i = 0;
            for(uint32_t id : idList)
                AddIdIndexPair(id, i);
        }
        else AddIdIndexPair(idList, 0);
    }
    else
    {
        // Determine if this node is a multinode
        auto it = jsn.cend();
        if((it = jsn.find(NodeNames::ID)) != jsn.cend())
        {
            isMultiNode = it->is_array();
        }
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

bool SceneNodeJson::CheckNode(const std::string& name) const
{
    nlohmann::json::const_iterator i;
    if((i = node.find(name)) != node.end())
        return true;
    return false;
}

size_t SceneNodeJson::CommonListSize(const std::string& name) const
{
    const nlohmann::json& nodeInner = node[name];
    return nodeInner.size();
}

bool SceneNodeJson::CommonBool(const std::string& name, double time) const
{
    return SceneIO::LoadBool(node[name], time);
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

std::vector<bool> SceneNodeJson::CommonBoolList(const std::string& name, double time) const
{
    return CommonList<bool, SceneIO::LoadBool>(name, time);
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
    result.reserve(idIndexPairs.size());

    for(const auto& list : idIndexPairs)
    {
        const InnerIndex i = list.second;
        const nlohmann::json& node = (idIndexPairs.size() > 1) ? nodeInner[i] : nodeInner;

        result.push_back(node.size());
    }
    return std::move(result);
}

std::vector<uint32_t> SceneNodeJson::AccessUIntRanged(const std::string& name) const
{
    return AccessRanged<uint32_t>(name);
}

std::vector<uint64_t> SceneNodeJson::AccessUInt64Ranged(const std::string& name) const
{
    return AccessRanged<uint64_t>(name);
}

std::vector<bool> SceneNodeJson::AccessBool(const std::string& name, double time) const
{
    return AccessSingle<bool, SceneIO::LoadBool>(name, time);
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

std::vector<BoolList> SceneNodeJson::AccessBoolList(const std::string& name, double time) const
{
    return AccessList<bool, SceneIO::LoadBool>(name, time);
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

std::vector<NodeTextureStruct> SceneNodeJson::AccessTextureNode(const std::string& name, double time) const
{
    return AccessSingle<NodeTextureStruct, SceneIO::LoadNodeTextureStruct>(name, time);
}

TexturedDataNodeList<float> SceneNodeJson::AccessTexturedDataFloat(const std::string& name, double time) const
{
    return AccessTextured<float, SceneIO::LoadNumber<float>>(name, time);
}

TexturedDataNodeList<Vector2> SceneNodeJson::AccessTexturedDataVector2(const std::string& name, double time) const
{
    return AccessTextured<Vector2, SceneIO::LoadVector<2, float>>(name, time);
}

TexturedDataNodeList<Vector3> SceneNodeJson::AccessTexturedDataVector3(const std::string& name, double time) const
{
    return AccessTextured<Vector3, SceneIO::LoadVector<3, float>>(name, time);
}

TexturedDataNodeList<Vector4> SceneNodeJson::AccessTexturedDataVector4(const std::string& name, double time) const
{
    return AccessTextured<Vector4, SceneIO::LoadVector<4, float>>(name, time);
}

OptionalNodeList<NodeTextureStruct> SceneNodeJson::AccessOptionalTextureNode(const std::string& name,
                                                                             double time) const
{
    return AccessOptional<NodeTextureStruct, SceneIO::LoadNodeTextureStruct>(name, time);
}