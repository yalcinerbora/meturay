#pragma once
/**

Have to wrap json lib since it does not work properly
with nvcc

*/

#include <cstdint>
#include <cassert>
#include <nlohmann/json_fwd.hpp>

#include "RayLib/SceneStructs.h"

struct SceneNodeI
{
    private:
        NodeId                          id;
        std::set<InnerIndex>            indices;

    protected:
    public:
        // Constructors & Destructor
                                            SceneNodeI();
                                            SceneNodeI(NodeId id);
                                            SceneNodeI(const SceneNodeI&) = default;
                                            SceneNodeI(SceneNodeI&&) = default;
        SceneNodeI&                         operator=(const SceneNodeI&) = delete;
        SceneNodeI&                         operator=(SceneNodeI&&) = default;
                                            ~SceneNodeI() = default;

        NodeId                              Id() const;
        const std::set<InnerIndex>&         Indices() const;
        size_t                              InnerIndexCount() const;
        void                                AddIndex(InnerIndex index);
        bool                                operator<(const SceneNodeI& node) const;

        // Interface
        virtual size_t                      AccessCount(uint32_t index) const = 0;
        virtual size_t                      AccessCount(const std::string& name) const = 0;

        virtual std::string                 AccessString(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<std::string>    AccessStringList(const std::string& name, double time = 0.0) const = 0;

        virtual float                       AccessFloat(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<float>          AccessFloatList(const std::string& name, double time) const = 0;

        virtual Vector2                     AccessVector2(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector2f>       AccessVector2List(uint32_t index, double time = 0.0) const = 0;
        virtual std::vector<Vector2f>       AccessVector2List(const std::string& name, double time = 0.0) const = 0;

        virtual Vector3                     AccessVector3(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector3f>       AccessVector3List(const std::string& name, double time = 0.0) const = 0;

        virtual Vector4                     AccessVector4(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Vector4>        AccessVector4List(const std::string& name, double time = 0.0) const = 0;

        virtual Matrix4x4                   AccessMatrix4x4(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<Matrix4x4>      AccessMatrix4x4List(const std::string& name, double time = 0.0) const = 0;

        virtual uint32_t                    AccessUInt(const std::string& name, double time = 0.0) const = 0;
        virtual std::vector<uint32_t>       AccessUIntList(const std::string& name, double time = 0.0) const = 0;
};

inline SceneNodeI::SceneNodeI()
    : SceneNodeI(0)
{}

inline SceneNodeI::SceneNodeI(NodeId id)
    : id(id)
{}

inline uint32_t SceneNodeI::Id() const
{
    return id;
}

inline const std::set<InnerIndex>& SceneNodeI::Indices() const
{
    return indices;
}

size_t SceneNodeI::InnerIndexCount() const
{
    return indices.size();
}

void SceneNodeI::AddIndex(InnerIndex iIndex)
{
    indices.emplace(iIndex);
}

inline bool SceneNodeI::operator<(const SceneNodeI& node) const
{
    return (id < node.id);
}