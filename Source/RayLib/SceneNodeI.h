#pragma once
/**

Have to wrap json lib since it does not work properly
with nvcc

*/

#include <memory>
#include <cstdint>
#include <cassert>

#include "RayLib/SceneStructs.h"

using BoolList = std::vector<bool>;
using StringList = std::vector<std::string>;
using FloatList = std::vector<float>;
using Vector2List = std::vector<Vector2>;
using Vector3List = std::vector<Vector3>;
using Vector4List = std::vector<Vector4>;
using Matrix4x4List = std::vector<Matrix4x4>;
using UIntList = std::vector<uint32_t>;
using UInt64List = std::vector<uint64_t>;
using TextureList = std::vector<NodeTextureStruct>;

template <class T>
using OptionalNode = std::pair<bool, T>;

template <class T>
struct TexturedDataNode
{
    bool isTexture;
    union
    {
        T data;
        NodeTextureStruct texNode;
    };
};

template <class T>
using OptionalNodeList = std::vector<OptionalNode<T>>;

template <class T>
using TexturedDataNodeList = std::vector<TexturedDataNode<T>>;

class SceneNodeI
{
    private:
    protected:
        bool                                isMultiNode;
        NodeIndex                           nodeIndex;
        std::set<IdPair>                    idIndexPairs;

    public:
        // Constructors & Destructor
                                            SceneNodeI() = delete;
                                            SceneNodeI(NodeIndex index, bool isMultiNode = false);
                                            SceneNodeI(const SceneNodeI&) = default;
                                            SceneNodeI(SceneNodeI&&) = default;
        SceneNodeI&                         operator=(const SceneNodeI&) = delete;
        SceneNodeI&                         operator=(SceneNodeI&&) = default;
        virtual                             ~SceneNodeI() = default;

        NodeIndex                           Index() const;
        const std::set<IdPair>&             Ids() const;
        size_t                              IdCount() const;
        void                                AddIdIndexPair(InnerIndex index, NodeId id);
        bool                                operator<(const SceneNodeI& node) const;

        // Interface
        virtual std::string                 Name() const = 0;
        virtual std::string                 Tag() const = 0;

        // Check availability of certain common / access node
        virtual bool                        CheckNode(const std::string& name) const = 0;

        // Id pair unspecific data loading
        virtual size_t                      CommonListSize(const std::string& name) const = 0;

        virtual bool                        CommonBool(const std::string& name, double time = 0.0) const = 0;
        virtual std::string                 CommonString(const std::string& name, double time = 0.0) const = 0;
        virtual float                       CommonFloat(const std::string& name, double time = 0.0) const = 0;
        virtual Vector2                     CommonVector2(const std::string& name, double time = 0.0) const = 0;
        virtual Vector3                     CommonVector3(const std::string& name, double time = 0.0) const = 0;
        virtual Vector4                     CommonVector4(const std::string& name, double time = 0.0) const = 0;
        virtual Matrix4x4                   CommonMatrix4x4(const std::string& name, double time = 0.0) const = 0;
        virtual uint32_t                    CommonUInt(const std::string& name, double time = 0.0) const = 0;
        virtual uint64_t                    CommonUInt64(const std::string& name, double time = 0.0) const = 0;

        virtual std::vector<bool>           CommonBoolList(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<std::string>    CommonStringList(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<float>          CommonFloatList(const std::string& name, double time) const = 0;
        virtual std::vector<Vector2>        CommonVector2List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector3>        CommonVector3List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector4>        CommonVector4List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Matrix4x4>      CommonMatrix4x4List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<uint32_t>       CommonUIntList(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<uint64_t>       CommonUInt64List(const std::string& name, double time = 0.0) const = 0;

        // Id pair specific data loading
        virtual size_t                      AccessListTotalCount(const std::string& name) const = 0;
        virtual std::vector<size_t>         AccessListCount(const std::string& name) const = 0;

        virtual std::vector<uint32_t>       AccessUIntRanged(const std::string& name) const = 0;
        virtual std::vector<uint64_t>       AccessUInt64Ranged(const std::string& name) const = 0;

        virtual std::vector<bool>           AccessBool(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<std::string>    AccessString(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<float>          AccessFloat(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector2>        AccessVector2(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector3>        AccessVector3(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector4>        AccessVector4(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Matrix4x4>      AccessMatrix4x4(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<uint32_t>       AccessUInt(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<uint64_t>       AccessUInt64(const std::string& name, double time = 0.0) const = 0;

        virtual std::vector<BoolList>       AccessBoolList(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<StringList>     AccessStringList(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<FloatList>      AccessFloatList(const std::string& name, double time) const = 0;
        virtual std::vector<Vector2List>    AccessVector2List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector3List>    AccessVector3List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector4List>    AccessVector4List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Matrix4x4List>  AccessMatrix4x4List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<UIntList>       AccessUIntList(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<UInt64List>     AccessUInt64List(const std::string& name, double time = 0.0) const = 0;

        virtual OptionalNodeList<bool>           AccessOptionalBool(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<float>          AccessOptionalFloat(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<Vector2>        AccessOptionalVector2(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<Vector3>        AccessOptionalVector3(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<Vector4>        AccessOptionalVector4(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<Matrix4x4>      AccessOptionalMatrix4x4(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<uint32_t>       AccessOptionalUInt(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<uint64_t>       AccessOptionalUInt64(const std::string& name, double time = 0.0) const = 0;

        virtual OptionalNodeList<BoolList>       AccessOptionalBoolList(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<FloatList>      AccessOptionalFloatList(const std::string& name, double time) const = 0;
        virtual OptionalNodeList<Vector2List>    AccessOptionalVector2List(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<Vector3List>    AccessOptionalVector3List(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<Vector4List>    AccessOptionalVector4List(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<Matrix4x4List>  AccessOptionalMatrix4x4List(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<UIntList>       AccessOptionalUIntList(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<UInt64List>     AccessOptionalUInt64List(const std::string& name, double time = 0.0) const = 0;

        // Texture Related
        virtual NodeTextureStruct               CommonTextureNode(const std::string& name, double time = 0.0) const = 0;
        virtual TexturedDataNode<float>         CommonTexturedDataFloat(const std::string& name, double time = 0.0) const = 0;
        virtual TexturedDataNode<Vector2>       CommonTexturedDataVector2(const std::string& name, double time = 0.0) const = 0;
        virtual TexturedDataNode<Vector3>       CommonTexturedDataVector3(const std::string& name, double time = 0.0) const = 0;
        virtual TexturedDataNode<Vector4>       CommonTexturedDataVector4(const std::string& name, double time = 0.0) const = 0;

        virtual std::vector<NodeTextureStruct>          AccessTextureNode(const std::string& name, double time = 0.0) const = 0;
        virtual TexturedDataNodeList<float>             AccessTexturedDataFloat(const std::string& name, double time = 0.0) const = 0;
        virtual TexturedDataNodeList<Vector2>           AccessTexturedDataVector2(const std::string& name, double time = 0.0) const = 0;
        virtual TexturedDataNodeList<Vector3>           AccessTexturedDataVector3(const std::string& name, double time = 0.0) const = 0;
        virtual TexturedDataNodeList<Vector4>           AccessTexturedDataVector4(const std::string& name, double time = 0.0) const = 0;
        virtual OptionalNodeList<NodeTextureStruct>     AccessOptionalTextureNode(const std::string& name, double time = 0.0) const = 0;
};

using SceneNodePtr = std::unique_ptr<SceneNodeI>;

inline SceneNodeI::SceneNodeI(NodeIndex index, bool isMultiNode)
    : isMultiNode(isMultiNode)
    , nodeIndex(index)
{}

inline uint32_t SceneNodeI::Index() const
{
    return nodeIndex;
}

inline const std::set<IdPair>& SceneNodeI::Ids() const
{
    return idIndexPairs;
}

inline size_t SceneNodeI::IdCount() const
{
    return idIndexPairs.size();
}

inline void SceneNodeI::AddIdIndexPair(NodeId id, InnerIndex index)
{
    idIndexPairs.emplace(std::make_pair(id, index));
}

inline bool SceneNodeI::operator<(const SceneNodeI& node) const
{
    return (nodeIndex < node.nodeIndex);
}