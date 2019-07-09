#pragma once
/**

Have to wrap json lib since it does not work properly
with nvcc

*/

#include <cstdint>
#include <cassert>
#include <nlohmann/json_fwd.hpp>

#include "RayLib/SceneStructs.h"

class SceneNodeI
{
    private:
    protected:
        NodeIndex                           nodeIndex;
        std::set<IdPair>                    indexIdPairs;

    public:
        // Constructors & Destructor
                                            SceneNodeI() = delete;
                                            SceneNodeI(NodeIndex index);
                                            SceneNodeI(const SceneNodeI&) = default;
                                            SceneNodeI(SceneNodeI&&) = default;
        SceneNodeI&                         operator=(const SceneNodeI&) = delete;
        SceneNodeI&                         operator=(SceneNodeI&&) = default;
        virtual                             ~SceneNodeI() = default;

        NodeIndex                           Index() const;
        const std::set<IdPair>&             Ids() const;
        size_t                              IdCount() const;
        void                                AddIndexIdPair(InnerIndex index, NodeId id);
        bool                                operator<(const SceneNodeI& node) const;

        // Interface
        virtual size_t                      AccessCount() const = 0;
        virtual size_t                      AccessCount(const std::string& name) const = 0;

        // Direct Singular data loading (id inspecific)
        virtual std::string                 AccessString(const std::string& name, double time = 0.0) const = 0;
        virtual float                       AccessFloat(const std::string& name, double time = 0.0) const = 0;
        virtual Vector2                     AccessVector2(const std::string& name, double time = 0.0) const = 0;
        virtual Vector3                     AccessVector3(const std::string& name, double time = 0.0) const = 0;
        virtual Vector4                     AccessVector4(const std::string& name, double time = 0.0) const = 0;
        virtual Matrix4x4                   AccessMatrix4x4(const std::string& name, double time = 0.0) const = 0;
        virtual uint32_t                    AccessUInt(const std::string& name, double time = 0.0) const = 0;

        // Id pair specific data loading
        virtual std::vector<std::string>    AccessStringList(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<float>          AccessFloatList(const std::string& name, double time) const = 0;
        virtual std::vector<Vector2>        AccessVector2List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector3>        AccessVector3List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector4>        AccessVector4List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Matrix4x4>      AccessMatrix4x4List(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<uint32_t>       AccessUIntList(const std::string& name, double time = 0.0) const = 0;
};

inline SceneNodeI::SceneNodeI(NodeIndex index)
    : nodeIndex(index)
{}

inline uint32_t SceneNodeI::Index() const
{
    return nodeIndex;
}

inline const std::set<IdPair>& SceneNodeI::Ids() const
{
    return indexIdPairs;
}

size_t SceneNodeI::IdCount() const
{
    return indexIdPairs.size();
}

void SceneNodeI::AddIndexIdPair(InnerIndex index, NodeId id)
{
    indexIdPairs.emplace(std::make_pair(index, id));
}

inline bool SceneNodeI::operator<(const SceneNodeI& node) const
{
    return (nodeIndex < node.nodeIndex);
}