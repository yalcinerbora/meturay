#pragma once

#include <string>

enum class LightType
{
    POINT,
    DIRECTIONAL,
    SPOT,
    RECTANGULAR,
    TRIANGULAR,
    DISK,
    SPHERICAL,
    PRIMITIVE,
    //
    END

};

static constexpr const char* LightTypeNames[static_cast<int>(LightType::END)] =
{
    "point",
    "directional",
    "spot",
    "rectangular",
    "triangular",
    "disk",
    "spherical",
    "primitive"
};

struct EstimatorInfo
{
    LightType   type;

    Vector4     position0X;
    Vector4     position1Y;
    Vector4     position2Z;
};


inline SceneError LightTypeStringToEnum(LightType& type,
                                        const std::string& str)
{
    for(int i = 0; i < static_cast<int>(LightType::END); i++)
    {
        if(str == std::string(LightTypeNames[i]))
        {
           type = static_cast<LightType>(i);
           return SceneError::OK;
        }
    }
    return SceneError::UNKNOWN_TRANSFORM_TYPE;
}

inline SceneError FetchLightInfoFromNode(EstimatorInfo&, const SceneNodeI& node,
                                         LightType type)
{
    // TODO:

    return SceneError::OK;
}
