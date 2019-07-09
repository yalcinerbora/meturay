#pragma once

#include <nlohmann/json_fwd.hpp>

#include "RayLib/SceneNodeI.h"

class SceneNodeJson final : public SceneNodeI
{
    private:
        nlohmann::json&              node;

    protected:
    public:
        // Constructors & Destructor
                                    SceneNodeJson(nlohmann::json&, NodeId id);
                                    ~SceneNodeJson() = default;

        // Interface
        size_t                      AccessCount() const override;
        size_t                      AccessCount(const std::string& name) const override;

        // Direct Singular data loading (id inspecific)
        std::string                 AccessString(const std::string& name, double time = 0.0) const override;
        float                       AccessFloat(const std::string& name, double time = 0.0) const override;
        Vector2                     AccessVector2(const std::string& name, double time = 0.0) const override;
        Vector3                     AccessVector3(const std::string& name, double time = 0.0) const override;
        Vector4                     AccessVector4(const std::string& name, double time = 0.0) const override;
        Matrix4x4                   AccessMatrix4x4(const std::string& name, double time = 0.0) const override;
        uint32_t                    AccessUInt(const std::string& name, double time = 0.0) const override;

        // Id pair specific data loading
        std::vector<std::string>    AccessStringList(const std::string& name, double time = 0.0) const override;
        std::vector<float>          AccessFloatList(const std::string& name, double time) const override;
        std::vector<Vector2>        AccessVector2List(const std::string& name, double time = 0.0) const override;
        std::vector<Vector3>        AccessVector3List(const std::string& name, double time = 0.0) const override;
        std::vector<Vector4>        AccessVector4List(const std::string& name, double time = 0.0) const override;
        std::vector<Matrix4x4>      AccessMatrix4x4List(const std::string& name, double time = 0.0) const override;
        std::vector<uint32_t>       AccessUIntList(const std::string& name, double time = 0.0) const override;
        
};
