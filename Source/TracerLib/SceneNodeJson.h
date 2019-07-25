#pragma once

#include <nlohmann/json_fwd.hpp>

#include "RayLib/SceneNodeI.h"

template <class T>
using LoadFunc = T(*)(const nlohmann::json&, double);

class SceneNodeJson final : public SceneNodeI
{
    private:
        const nlohmann::json&       node;

        template <class T, LoadFunc<T>>
        std::vector<std::vector<T>>     AccessList(const nlohmann::json& node,
                                                   const std::string& name,
                                                   double time) const;
        template <class T, LoadFunc<T>>
        std::vector<T>                  AccessSingle(const nlohmann::json& node,
                                                     const std::string& name,
                                                     double time) const;

    protected:
    public:
        // Constructors & Destructor
                                    SceneNodeJson(const nlohmann::json&, NodeId id);
                                    ~SceneNodeJson() = default;

        // Interface
        size_t                      AccessListTotalCount(const std::string& name) const override;
        std::vector<size_t>         AccessListCount(const std::string& name) const override;

        // Direct Singular data loading (id inspecific)
        std::string                 Name() const override;
        std::string                 Tag() const override;

        // Id pair specific data loading
        std::vector<std::string>    AccessString(const std::string& name, double time = 0.0) const override;
        std::vector<float>          AccessFloat(const std::string& name, double time = 0.0) const override;
        std::vector<Vector2>        AccessVector2(const std::string& name, double time = 0.0) const override;
        std::vector<Vector3>        AccessVector3(const std::string& name, double time = 0.0) const override;
        std::vector<Vector4>        AccessVector4(const std::string& name, double time = 0.0) const override;
        std::vector<Matrix4x4>      AccessMatrix4x4(const std::string& name, double time = 0.0) const override;
        std::vector<uint32_t>       AccessUInt(const std::string& name, double time = 0.0) const override;

        std::vector<StringList>     AccessStringList(const std::string& name, double time = 0.0) const override;
        std::vector<FloatList>      AccessFloatList(const std::string& name, double time) const override;
        std::vector<Vector2List>    AccessVector2List(const std::string& name, double time = 0.0) const override;
        std::vector<Vector3List>    AccessVector3List(const std::string& name, double time = 0.0) const override;
        std::vector<Vector4List>    AccessVector4List(const std::string& name, double time = 0.0) const override;
        std::vector<Matrix4x4List>  AccessMatrix4x4List(const std::string& name, double time = 0.0) const override;
        std::vector<UIntList>       AccessUIntList(const std::string& name, double time = 0.0) const override;
        
};
