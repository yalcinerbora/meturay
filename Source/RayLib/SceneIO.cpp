#include "SceneIO.h"
#include "Camera.h"
#include "Constants.h"
#include "SceneNodeNames.h"

using namespace NodeNames;

TransformStruct SceneIO::LoadTransform(const nlohmann::json& jsn, double time)
{
    if(jsn.is_string())
    {
        return LoadFromAnim<TransformStruct>(jsn, time);
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

CameraPerspective SceneIO::LoadCamera(const nlohmann::json& jsn, double time)
{
    if(jsn.is_string())
    {
        return LoadFromAnim<CameraPerspective>(jsn, time);
    }
    else if(jsn.is_object())
    {
        CameraPerspective cam = {};
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

LightStruct SceneIO::LoadLight(const nlohmann::json& jsn, double time)
{
    if(jsn.is_string())
    {
        return LoadFromAnim<LightStruct>(jsn, time);
    }
    else if(jsn.is_object())
    {
        std::string type = jsn[TYPE];
        uint32_t matId = jsn[LIGHT_MATERIAL];
     
        return LightStruct{type, matId};   
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