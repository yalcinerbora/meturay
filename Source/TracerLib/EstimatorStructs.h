#pragma once

#include <string>

#include "RayLib/SceneNodeNames.h"
#include "RayLib/HitStructs.h"
#include "RayLib/SceneError.h"

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
    HitKey      matKey;    
    Vector4     position0R;
    Vector4     position1G;
    Vector4     position2B;

    static EstimatorInfo GenOnlyPower(const Vector3& flux);
    static EstimatorInfo GenAsPoint(const HitKey key, 
                                    const Vector3& flux,
                                    const Vector3& position);
    static EstimatorInfo GenAsDirectional(const HitKey key,
                                          const Vector3& flux,
                                          const Vector3& direction);
    static EstimatorInfo GenAsSpot(const HitKey key, 
                                   const Vector3& flux,
                                   const Vector3& position,
                                   const Vector3& direction,
                                   float coneMin, float coneMax);
    static EstimatorInfo GenAsRectangular(const HitKey key, 
                                          const Vector3& flux,
                                          const Vector3& topLeft,
                                          const Vector3& v0,
                                          const Vector3& v1);
    static EstimatorInfo GenAsTriangular(const HitKey key, 
                                         const Vector3& flux,
                                         const Vector3& position0,
                                         const Vector3& position1,
                                         const Vector3& position2);
    static EstimatorInfo GenAsDisk(const HitKey key, 
                                   const Vector3& flux,
                                   const Vector3& center,
                                   const Vector3& normal,
                                   float radius);
    static EstimatorInfo GenAsSpherical(const HitKey key, 
                                        const Vector3& flux,
                                        const Vector3& center,
                                        float radius);
};

inline EstimatorInfo EstimatorInfo::GenOnlyPower(const Vector3& flux)
{
    EstimatorInfo r;
    r.position0R[3] = flux[0];
    r.position1G[3] = flux[1];
    r.position2B[3] = flux[2];
    return r;
}

inline EstimatorInfo EstimatorInfo::GenAsPoint(const HitKey key, 
                                               const Vector3& flux,
                                               const Vector3& position)
{
    EstimatorInfo r;
    r.matKey = key;
    r.type = LightType::POINT;
    r.position0R = Vector4(position, flux[0]);
    r.position1G = Vector4(Zero3, flux[1]);
    r.position2B = Vector4(Zero3, flux[2]);
    return r;
}

inline EstimatorInfo EstimatorInfo::GenAsDirectional(const HitKey key, 
                                                     const Vector3& flux,
                                                     const Vector3& direction)
{
    EstimatorInfo r;
    r.matKey = key;
    r.type = LightType::DIRECTIONAL;
    r.position0R = Vector4(direction, flux[0]);
    r.position1G = Vector4(Zero3, flux[1]);
    r.position2B = Vector4(Zero3, flux[2]);
    return r;
}

inline EstimatorInfo EstimatorInfo::GenAsSpot(const HitKey key, 
                                              const Vector3& flux,
                                              const Vector3& position,
                                              const Vector3& direction,
                                              float coneMin, float coneMax)
{
    EstimatorInfo r;
    r.matKey = key;
    r.type = LightType::SPOT;
    r.position0R = Vector4(position, flux[0]);
    r.position1G = Vector4(direction, flux[1]);
    r.position2B = Vector4(coneMin, coneMax, 0.0f, flux[2]);
    return r;
}

inline EstimatorInfo EstimatorInfo::GenAsRectangular(const HitKey key, 
                                                     const Vector3& flux,
                                                     const Vector3& topLeft,
                                                     const Vector3& v0,
                                                     const Vector3& v1)
{    
    EstimatorInfo r;
    r.matKey = key;
    r.type = LightType::RECTANGULAR;
    r.position0R = Vector4(topLeft, flux[0]);
    r.position1G = Vector4(v0, flux[1]);
    r.position2B = Vector4(v1, flux[2]);
    return r;
}

inline EstimatorInfo EstimatorInfo::GenAsTriangular(const HitKey key, 
                                                    const Vector3& flux,
                                                    const Vector3& position0,
                                                    const Vector3& position1,
                                                    const Vector3& position2)
{
    EstimatorInfo r;
    r.matKey = key;
    r.type = LightType::TRIANGULAR;
    r.position0R = Vector4(position0, flux[0]);
    r.position1G = Vector4(position1, flux[1]);
    r.position2B = Vector4(position2, flux[2]);
    return r;
}

inline EstimatorInfo EstimatorInfo::GenAsDisk(const HitKey key, 
                                              const Vector3& flux,
                                              const Vector3& center,
                                              const Vector3& direction,
                                              float radius)
{
    EstimatorInfo r;
    r.matKey = key;
    r.type = LightType::DISK;
    r.position0R = Vector4(center, flux[0]);
    r.position1G = Vector4(direction, flux[1]);
    r.position2B = Vector4(radius, 0.0f, 0.0f, flux[2]);
    return r;
}

inline EstimatorInfo EstimatorInfo::GenAsSpherical(const HitKey key, 
                                                   const Vector3& flux,
                                                   const Vector3& center,
                                                   float radius)
{
    EstimatorInfo r;
    r.matKey = key;
    r.type = LightType::SPHERICAL;
    r.position0R = Vector4(center, flux[0]);
    r.position1G = Vector4(radius, 0.0f, 0.0f, flux[1]);
    r.position2B = Vector4(Zero3, flux[2]);
    return r;
}

inline SceneError LightTypeEnumToString(std::string& str,
                                        LightType type)
{
    int index = static_cast<int>(type);
    if(index >= static_cast<int>(LightType::END))
        return SceneError::UNKNOWN_LIGHT_TYPE;
    else str = LightTypeNames[index];
    return SceneError::OK;
}

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
    return SceneError::UNKNOWN_LIGHT_TYPE;
}

inline SceneError FetchLightInfoFromNode(std::vector<EstimatorInfo>& result, 
                                         const SceneNodeI& node,
                                         const MaterialKeyListing& materialKeys,
                                         LightType type, double time)
{
    using namespace NodeNames;
    using namespace BaseConstants;

    const auto fluxList = node.AccessVector3(LIGHT_POWER, time);
    const auto matIds = node.AccessUInt(LIGHT_MATERIAL);

    switch(type)
    {
        // Fetch from Node if analytic light
        case LightType::POINT:
        {
            const auto posList = node.AccessVector3(LIGHT_POSITION, time);
            for(size_t i = 0; i < fluxList.size(); i++)
            {
                const Vector3& pos = posList[i];
                const Vector3& flux = fluxList[i];
                const auto matLookup = std::make_pair(EMPTY_PRIMITIVE_NAME, matIds[i]);
                const HitKey key = materialKeys.at(matLookup);
                result.push_back(EstimatorInfo::GenAsPoint(key, flux, pos));
            }
            break;
        }
        case LightType::DIRECTIONAL:
        {
            const auto dirList = node.AccessVector3(LIGHT_DIRECTION, time);
            for(size_t i = 0; i < fluxList.size(); i++)
            {
                const Vector3& dir = dirList[i];
                const Vector3& flux = fluxList[i];
                const auto matLookup = std::make_pair(EMPTY_PRIMITIVE_NAME, matIds[i]);
                const HitKey key = materialKeys.at(matLookup);
                result.push_back(EstimatorInfo::GenAsDirectional(key, flux, dir));
            }
            break;
        }
        case LightType::SPOT:
        {
            const auto posList = node.AccessVector3(LIGHT_POSITION, time);
            const auto dirList = node.AccessVector3(LIGHT_DIRECTION, time);
            const auto angleList = node.AccessVector2(LIGHT_CONE_APERTURE, time);
            for(size_t i = 0; i < fluxList.size(); i++)
            {
                const Vector3& pos = posList[i];
                const Vector3& dir = dirList[i];
                const Vector2& angle = angleList[i];
                const Vector3& flux = fluxList[i];
                const auto matLookup = std::make_pair(EMPTY_PRIMITIVE_NAME, matIds[i]);
                const HitKey key = materialKeys.at(matLookup);
                result.push_back(EstimatorInfo::GenAsSpot(key, flux, pos, dir, 
                                                          angle[0], angle[1]));
            }
            break;
        }
        case LightType::RECTANGULAR:
        {
            const auto posList = node.AccessVector3(LIGHT_POSITION, time);
            const auto v0List = node.AccessVector3(LIGHT_RECT_V0, time);
            const auto v1List = node.AccessVector3(LIGHT_RECT_V1, time);
            for(size_t i = 0; i < fluxList.size(); i++)
            {
                const Vector3& pos = posList[i];
                const Vector3& v0 = v0List[i];
                const Vector3& v1 = v1List[i];
                const Vector3& flux = fluxList[i];
                const auto matLookup = std::make_pair(EMPTY_PRIMITIVE_NAME, matIds[i]);
                const HitKey key = materialKeys.at(matLookup);
                result.push_back(EstimatorInfo::GenAsRectangular(key, flux, pos, v0, v1));
            }
            break;
        }
        case LightType::TRIANGULAR:
        {
            const auto matIds = node.AccessUInt(LIGHT_MATERIAL);
            const auto posList = node.AccessVector3(LIGHT_POSITION, time);
            for(size_t i = 0; i < fluxList.size(); i++)
            {
                const Vector3& pos0 = posList[i * 3 + 0];
                const Vector3& pos1 = posList[i * 3 + 1];
                const Vector3& pos2 = posList[i * 3 + 2];
                const Vector3& flux = fluxList[i];
                const auto matLookup = std::make_pair(EMPTY_PRIMITIVE_NAME, matIds[i]);
                const HitKey key = materialKeys.at(matLookup);

                result.push_back(EstimatorInfo::GenAsTriangular(key, flux, pos0, pos1, pos2));
            }
            break;
        }
        case LightType::DISK:
        {
            const auto dirList = node.AccessVector3(LIGHT_DIRECTION, time);
            const auto centerList = node.AccessVector3(LIGHT_DISK_CENTER, time);
            const auto radiusList = node.AccessFloat(LIGHT_DISK_RADIUS, time);
            for(size_t i = 0; i < fluxList.size(); i++)
            {
                const Vector3& center = centerList[i];
                const Vector3& dir = dirList[i];
                const float& radius = radiusList[i];
                const Vector3& flux = fluxList[i];
                const auto matLookup = std::make_pair(EMPTY_PRIMITIVE_NAME, matIds[i]);
                const HitKey key = materialKeys.at(matLookup);
                result.push_back(EstimatorInfo::GenAsDisk(key, flux, center, dir, radius));
            }
            break;
        }
        case LightType::SPHERICAL:
        {
            const auto centerList = node.AccessVector3(LIGHT_SPHR_CENTER, time);
            const auto radiusList = node.AccessFloat(LIGHT_SPHR_RADIUS, time);
            for(size_t i = 0; i < fluxList.size(); i++)
            {
                const Vector3& center = centerList[i];
                const float& radius = radiusList[i];
                const Vector3& flux = fluxList[i];
                const auto matLookup = std::make_pair(EMPTY_PRIMITIVE_NAME, matIds[i]);
                const HitKey key = materialKeys.at(matLookup);
                result.push_back(EstimatorInfo::GenAsSpherical(key, flux, center, radius));
            }
            break;
        }
        case LightType::PRIMITIVE: { break; }
        default: return SceneError::UNKNOWN_LIGHT_TYPE;
    }
    return SceneError::OK;
}
