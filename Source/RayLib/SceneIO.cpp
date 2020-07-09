#include "SceneIO.h"
#include "Camera.h"
#include "Constants.h"
#include "SceneNodeNames.h"

using namespace NodeNames;

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

static constexpr const char* TextureTypeNames[static_cast<int>(TextureType::END)] =
{
    "1D",
    "2D",
    "3D",
    "cube"
};

static constexpr const char* FilterTypeNames[static_cast<int>(FilterType::END)] =
{
    "linear",
    "nearest"
};

inline SceneError TextureTypeStringToEnum(TextureType& type,
                                          const std::string& str)
{
    for(int i = 0; i < static_cast<int>(TextureType::END); i++)
    {
        if(str == std::string(TextureTypeNames[i]))
        {
            type = static_cast<TextureType>(i);
            return SceneError::OK;
        }
    }
    return SceneError::UNKNOWN_TEXTURE_TYPE;
}

inline SceneError FilterTypeStringToEnum(FilterType& type,
                                          const std::string& str)
{
    for(int i = 0; i < static_cast<int>(FilterType::END); i++)
    {
        if(str == std::string(FilterTypeNames[i]))
        {
            type = static_cast<FilterType>(i);
            return SceneError::OK;
        }
    }
    return SceneError::UNKNOWN_FILTER_TYPE;
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

GPUTransform SceneIO::LoadTransform(const nlohmann::json& jsn, double time)
{
    if(jsn.is_string())
    {
        return LoadFromAnim<GPUTransform>(jsn, time);
    }
    else if(jsn.is_object())
    {
        std::string type = LoadString(jsn[TYPE], time);
        if(type == TRANSFORM_FORM_MATRIX4)
        {
            Matrix4x4 mat = LoadMatrix<4, float>(jsn[DATA], time);
            return mat;
        }
        else if(type == TRANSFORM_FORM_T_R_S)
        {
            Vector3 translation = LoadVector<3, float>(jsn[DATA][0], time);
            Vector3 rotation = LoadVector<3, float>(jsn[DATA][1], time);
            Vector3 scale = LoadVector<3, float>(jsn[DATA][2], time);

            Matrix4x4 mat = TransformGen::Translate(translation);
            mat *= TransformGen::Scale(scale[0], scale[1], scale[2]);
            mat *= TransformGen::Rotate(rotation[2] * MathConstants::DegToRadCoef, ZAxis);
            mat *= TransformGen::Rotate(rotation[1] * MathConstants::DegToRadCoef, YAxis);
            mat *= TransformGen::Rotate(rotation[0] * MathConstants::DegToRadCoef, XAxis);

            return mat;
        }
        else throw SceneException(SceneError::UNKNOWN_TRANSFORM_TYPE);

    }
    else throw SceneException(SceneError::TYPE_MISMATCH);
}

CPUCamera SceneIO::LoadCamera(const nlohmann::json& jsn, double time)
{
    if(jsn.is_string())
    {
        return LoadFromAnim<CPUCamera>(jsn, time);
    }
    else if(jsn.is_object())
    {
        CPUCamera cam = {};
        cam.position = LoadVector<3, float>(jsn[POSITION], time);
        cam.up = LoadVector<3, float>(jsn[CAMERA_UP], time);
        cam.gazePoint = LoadVector<3, float>(jsn[CAMERA_GAZE], time);
        Vector2 planes = LoadVector<2, float>(jsn[CAMERA_PLANES], time);
        cam.nearPlane = planes[0];
        cam.farPlane = planes[1];
        cam.fov = LoadVector<2, float>(jsn[CAMERA_FOV], time);
        cam.apertureSize = LoadNumber<float>(jsn[CAMERA_APERTURE], time);

        // Convert FOV to Radians
        cam.fov *= MathConstants::DegToRadCoef;

        return cam;
    }
    else throw SceneException(SceneError::TYPE_MISMATCH);
}

uint32_t SceneIO::LoadLightMatId(const nlohmann::json& jsn)
{
    return jsn[LIGHT_MATERIAL];
}

LightType SceneIO::LoadLightType(const nlohmann::json& jsn)
{
    LightType t;
    std::string type = jsn[TYPE];
    SceneError e = LightTypeStringToEnum(t, type);
    if(e != SceneError::OK) throw SceneException(SceneError::UNKNOWN_LIGHT_TYPE);
    return t;
}

CPULight SceneIO::LoadLight(const nlohmann::json& jsn, double time)
{
    if(jsn.is_string())
    {
        return LoadFromAnim<CPULight>(jsn, time);
    }
    else if(jsn.is_object())
    {
        CPULight light = CPULight{};
        light.position0 = Zero3;
        light.position1 = Zero3;
        light.position2 = Zero3;
        light.primId = 0;

        std::string type = jsn[TYPE];
        SceneError e = LightTypeStringToEnum(light.type, type);
        if(e != SceneError::OK) throw SceneException(SceneError::UNKNOWN_LIGHT_TYPE);

        light.flux = LoadVector<3, float>(jsn[LIGHT_POWER], time);
        switch(light.type)
        {
            // Fetch from Node if analytic light
            case LightType::POINT:
            {
                light.position0 = LoadVector<3, float>(jsn[LIGHT_POSITION], time);
                break;
            }
            case LightType::DIRECTIONAL:
            {
                light.position0 = LoadVector<3, float>(jsn[LIGHT_DIRECTION], time);
                break;
            }
            case LightType::SPOT:
            {
                light.position0 = LoadVector<3, float>(jsn[LIGHT_POSITION], time);
                light.position1 = LoadVector<3, float>(jsn[LIGHT_DIRECTION], time);
                Vector2 angles = LoadVector<2, float>(jsn[LIGHT_CONE_APERTURE], time);
                angles *= MathConstants::DegToRadCoef;
                angles[0] = std::cos(angles[0]);
                angles[1] = std::cos(angles[1]);
                light.position2 = Vector3(angles, 0.0f);
                break;
            }
            case LightType::RECTANGULAR:
            {
                light.position0 = LoadVector<3, float>(jsn[LIGHT_POSITION], time);
                light.position1 = LoadVector<3, float>(jsn[LIGHT_RECT_V0], time);
                light.position2 = LoadVector<3, float>(jsn[LIGHT_RECT_V1], time);
                break;
            }
            case LightType::TRIANGULAR:
            {
                light.position0 = LoadVector<3, float>(jsn[LIGHT_POSITION][0], time);
                light.position1 = LoadVector<3, float>(jsn[LIGHT_POSITION][1], time);
                light.position2 = LoadVector<3, float>(jsn[LIGHT_POSITION][2], time);
                break;
            }
            case LightType::DISK:
            {
                light.position0 = LoadVector<3, float>(jsn[LIGHT_DIRECTION], time);
                light.position1 = LoadVector<3, float>(jsn[LIGHT_POSITION], time);
                light.position2[0] = LoadNumber<float>(jsn[LIGHT_DISK_RADIUS], time);
                break;
            }
            case LightType::SPHERICAL:
            {
                light.position0 = LoadVector<3, float>(jsn[LIGHT_SPHR_CENTER], time);
                light.position1[0] = LoadNumber<float>(jsn[LIGHT_SPHR_RADIUS], time);
                break;
            }
            // Skip primitive ones
            case LightType::PRIMITIVE: { break; }
            default: throw SceneException(SceneError::UNKNOWN_LIGHT_TYPE);
        }
        return light;
    }
    else throw SceneException(SceneError::TYPE_MISMATCH);
}

SurfaceStruct SceneIO::LoadSurface(const nlohmann::json& jsn, double time)
{
    if(jsn.is_string())
    {
        return LoadFromAnim<SurfaceStruct>(jsn, time);
    }
    else
    {
        SurfaceStruct s = {};
        s.transformId = jsn[TRANSFORM];
        s.acceleratorId = jsn[ACCELERATOR];
        s.matPrimPairs.fill(std::make_pair(std::numeric_limits<uint32_t>::max(),
                                           std::numeric_limits<uint32_t>::max()));

        // Array Like Couples
        const auto primIdArray = jsn[PRIMITIVE];
        const auto materialIdArray = jsn[MATERIAL];
        if(primIdArray.size() != materialIdArray.size())
            throw SceneException(SceneError::PRIM_MATERIAL_NOT_SAME_SIZE);
        if(primIdArray.size() >= SceneConstants::MaxPrimitivePerSurface)
            throw SceneException(SceneError::TOO_MANY_SURFACE_ON_NODE);

        if(primIdArray.size() == 1)
        {
            s.matPrimPairs[0] = std::make_pair(materialIdArray, primIdArray);
        }
        else for(int i = 0; i < static_cast<int>(primIdArray.size()); i++)
            s.matPrimPairs[i] = std::make_pair(materialIdArray[i], primIdArray[i]);

        std::sort(s.matPrimPairs.begin(), s.matPrimPairs.end());
        s.pairCount = static_cast<int8_t>(primIdArray.size());
        return s;
    }
}

TextureStruct SceneIO::LoadTexture(const nlohmann::json& jsn, double time)
{
    if(jsn.is_string())
    {
        return LoadFromAnim<TextureStruct>(jsn, time);
    }
    else
    {
        TextureStruct s = {};
        s.id = jsn[ID];
        s.name = jsn[NAME];
        s.cached = jsn[TEXTURE_IS_CACHED];

        std::string typeName = jsn[TYPE];
        SceneError e = TextureTypeStringToEnum(s.type, typeName);
        if(e != SceneError::OK) throw SceneException(e);

        std::string filterName = jsn[TEXTURE_FILTER];
        e = FilterTypeStringToEnum(s.filter, filterName);
        if(e != SceneError::OK) throw SceneException(e);
        return s;
    }
}