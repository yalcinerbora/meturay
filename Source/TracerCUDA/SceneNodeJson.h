#pragma once

#include <nlohmann/json_fwd.hpp>

#include "RayLib/SceneNodeI.h"

template <class T>
using LoadFunc = T(*)(const nlohmann::json&, double);

class SceneNodeJson final : public SceneNodeI
{
    private:
        const nlohmann::json&           node;

        template <class T, LoadFunc<T>>
        std::vector<std::vector<T>>     AccessList(const std::string& name,
                                                   double time) const;
        template <class T, LoadFunc<T>>
        std::vector<T>                  AccessSingle(const std::string& name,
                                                     double time) const;
        template <class T>
        std::vector<T>                  AccessRanged(const std::string& name) const;
        template <class T, LoadFunc<T>>
        std::vector<T>                  CommonList(const std::string& name,
                                                   double time) const;
        template <class T, LoadFunc<T>>
        TexturedDataNodeList<T>         AccessTextured(const std::string& name,
                                                       double time) const;
        template <class T, LoadFunc<T>>
        OptionalNodeList<T>             AccessOptional(const std::string& name,
                                                       double time) const;

    protected:
    public:
        // Constructors & Destructor
                                        SceneNodeJson(const nlohmann::json&, NodeId id, bool forceFetchAll = false);
                                        ~SceneNodeJson() = default;

        // Direct Singular data loading (id inspecific)
        std::string                     Name() const override;
        std::string                     Tag() const override;

        // Check availability of certain node
        bool                            CheckNode(const std::string& name) const override;

        // Id pair inspecific data loading
        size_t                          CommonListSize(const std::string& name) const override;

        bool                            CommonBool(const std::string& name, double time = 0.0) const override;
        std::string                     CommonString(const std::string& name, double time = 0.0) const override;
        float                           CommonFloat(const std::string& name, double time = 0.0) const override;
        Vector2                         CommonVector2(const std::string& name, double time = 0.0) const override;
        Vector3                         CommonVector3(const std::string& name, double time = 0.0) const override;
        Vector4                         CommonVector4(const std::string& name, double time = 0.0) const override;
        Matrix4x4                       CommonMatrix4x4(const std::string& name, double time = 0.0) const override;
        uint32_t                        CommonUInt(const std::string& name, double time = 0.0) const override;
        uint64_t                        CommonUInt64(const std::string& name, double time = 0.0) const override;

        std::vector<bool>               CommonBoolList(const std::string& name, double time = 0.0) const override;
        std::vector<std::string>        CommonStringList(const std::string& name, double time = 0.0) const override;
        std::vector<float>              CommonFloatList(const std::string& name, double time) const override;
        std::vector<Vector2>            CommonVector2List(const std::string& name, double time = 0.0) const override;
        std::vector<Vector3>            CommonVector3List(const std::string& name, double time = 0.0) const override;
        std::vector<Vector4>            CommonVector4List(const std::string& name, double time = 0.0) const override;
        std::vector<Matrix4x4>          CommonMatrix4x4List(const std::string& name, double time = 0.0) const override;
        std::vector<uint32_t>           CommonUIntList(const std::string& name, double time = 0.0) const override;
        std::vector<uint64_t>           CommonUInt64List(const std::string& name, double time = 0.0) const override;

        // Id pair specific data loading
        size_t                          AccessListTotalCount(const std::string& name) const override;
        std::vector<size_t>             AccessListCount(const std::string& name) const override;

        // Special Integer Ranged Based Info
        std::vector<uint32_t>           AccessUIntRanged(const std::string& name) const override;
        std::vector<uint64_t>           AccessUInt64Ranged(const std::string& name) const override;

        std::vector<bool>               AccessBool(const std::string& name, double time = 0.0) const override;
        std::vector<std::string>        AccessString(const std::string& name, double time = 0.0) const override;
        std::vector<float>              AccessFloat(const std::string& name, double time = 0.0) const override;
        std::vector<Vector2>            AccessVector2(const std::string& name, double time = 0.0) const override;
        std::vector<Vector3>            AccessVector3(const std::string& name, double time = 0.0) const override;
        std::vector<Vector4>            AccessVector4(const std::string& name, double time = 0.0) const override;
        std::vector<Matrix4x4>          AccessMatrix4x4(const std::string& name, double time = 0.0) const override;
        std::vector<uint32_t>           AccessUInt(const std::string& name, double time = 0.0) const override;
        std::vector<uint64_t>           AccessUInt64(const std::string& name, double time = 0.0) const override;

        std::vector<BoolList>           AccessBoolList(const std::string& name, double time = 0.0) const override;
        std::vector<StringList>         AccessStringList(const std::string& name, double time = 0.0) const override;
        std::vector<FloatList>          AccessFloatList(const std::string& name, double time) const override;
        std::vector<Vector2List>        AccessVector2List(const std::string& name, double time = 0.0) const override;
        std::vector<Vector3List>        AccessVector3List(const std::string& name, double time = 0.0) const override;
        std::vector<Vector4List>        AccessVector4List(const std::string& name, double time = 0.0) const override;
        std::vector<Matrix4x4List>      AccessMatrix4x4List(const std::string& name, double time = 0.0) const override;
        std::vector<UIntList>           AccessUIntList(const std::string& name, double time = 0.0) const override;
        std::vector<UInt64List>         AccessUInt64List(const std::string& name, double time = 0.0) const override;

        // Texture Related
        std::vector<MaterialTextureStruct>      AccessTextureNode(const std::string& name,
                                                                  double time = 0.0) const override;
        TexturedDataNodeList<float>             AccessTexturedDataFloat(const std::string& name,
                                                                        double time = 0.0) const override;
        TexturedDataNodeList<Vector2>           AccessTexturedDataVector2(const std::string& name,
                                                                          double time = 0.0) const override;
        TexturedDataNodeList<Vector3>           AccessTexturedDataVector3(const std::string& name,
                                                                          double time = 0.0) const override;
        TexturedDataNodeList<Vector4>           AccessTexturedDataVector4(const std::string& name,
                                                                          double time = 0.0) const override;
        OptionalNodeList<MaterialTextureStruct> AccessOptionalTextureNode(const std::string& name,
                                                                          double time = 0.0) const override;
};
