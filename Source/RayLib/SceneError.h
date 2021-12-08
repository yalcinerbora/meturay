#pragma once
#pragma once
/**

Tracer error "Enumeration"

*/

#include <stdexcept>
#include "Error.h"

struct SceneError : public ErrorI
{
    public:
        enum Type
        {
            OK,
            // Common
            FILE_NOT_FOUND,
            ANIMATION_FILE_NOT_FOUND,
            // Not Found
            MATERIALS_ARRAY_NOT_FOUND,
            PRIMITIVES_ARRAY_NOT_FOUND,
            SURFACES_ARRAY_NOT_FOUND,
            ACCELERATORS_ARRAY_NOT_FOUND,
            LIGHTS_ARRAY_NOT_FOUND,
            TRANSFORMS_ARRAY_NOT_FOUND,
            CAMERAS_ARRAY_NOT_FOUND,
            MEDIUM_ARRAY_NOT_FOUND,
            BASE_ACCELERATOR_NODE_NOT_FOUND,
            BASE_BOUND_LIGHT_NODE_NOT_FOUND,
            BASE_MEDIUM_NODE_NOT_FOUND,
            LIGHT_SURFACES_ARRAY_NOT_FOUND,
            CAMERA_SURFACES_ARRAY_NOT_FOUND,
            // No Logic
            NO_LOGIC_FOR_ACCELERATOR,
            NO_LOGIC_FOR_MATERIAL,
            NO_LOGIC_FOR_PRIMITIVE,
            NO_LOGIC_FOR_SURFACE_DATA,
            NO_LOGIC_FOR_TRANSFORM,
            NO_LOGIC_FOR_MEDIUM,
            NO_LOGIC_FOR_CAMERA,
            NO_LOGIC_FOR_LIGHT,
            NO_LOGIC_FOR_TRACER,
            // Id Errors
            DUPLICATE_ACCELERATOR_ID,
            DUPLICATE_MATERIAL_ID,
            DUPLICATE_PRIMITIVE_ID,
            DUPLICATE_TRANSFORM_ID,
            DUPLICATE_MEDIUM_ID,
            DUPLICATE_LIGHT_ID,
            DUPLICATE_CAMERA_ID,
            // Id not found
            ACCELERATOR_ID_NOT_FOUND,
            MATERIAL_ID_NOT_FOUND,
            PRIMITIVE_ID_NOT_FOUND,
            TRANSFORM_ID_NOT_FOUND,
            MEDIUM_ID_NOT_FOUND,
            LIGHT_ID_NOT_FOUND,
            CAMERA_ID_NOT_FOUND,
            TEXTURE_ID_NOT_FOUND,
            // Json parse errors
            LOGIC_MISMATCH,
            TYPE_MISMATCH,
            JSON_FILE_PARSE_ERROR,
            // Special Type Values
            UNKNOWN_TRANSFORM_TYPE,
            UNKNOWN_LIGHT_TYPE,
            // Loading Surface Data
            SURFACE_DATA_PRIMITIVE_MISMATCH,
            SURFACE_DATA_TYPE_NOT_FOUND,
            SURFACE_DATA_INVALID_READ,
            // Some Mat/Accel Logic
            // may not support certain prims
            PRIM_ACCEL_MISMATCH,
            PRIM_MAT_MISMATCH,
            // Updating the Scene
            // Primitive Update Size Mismatch
            PRIM_UPDATE_SIZE_MISMATCH,
            // Too many types than key system can handle
            TOO_MANY_ACCELERATOR_GROUPS,
            TOO_MANY_ACCELERATOR_IN_GROUP,
            TOO_MANY_MATERIAL_GROUPS,
            TOO_MANY_MATERIAL_IN_GROUP,
            // Texture Related
            UNKNOWN_TEXTURE_ACCESS_LAYOUT,
            UNKNOWN_TEXTURE_TYPE,
            UNABLE_TO_LOAD_TEXTURE,
            TEXTURE_DIMENSION_MISMATCH,
            TEXTURE_CHANNEL_MISMATCH,
            TEXTURE_NOT_FOUND,
            BITMAP_LOAD_CALLED_WITH_MULTIPLE_CHANNELS,
            // Misc
            TOO_MANY_SURFACE_ON_NODE,
            PRIM_MATERIAL_NOT_SAME_SIZE,
            PRIM_TYPE_NOT_CONSISTENT_ON_SURFACE,
            OVERLAPPING_LIGHT_FOUND,
            OVERLAPPING_CAMERA_FOUND,
            PRIM_BACKED_LIGHT_AS_BOUNDARY,
            // Internal Errors
            INTERNAL_DUPLICATE_MAT_ID,
            INTERNAL_DUPLICATE_ACCEL_ID,
            //
            PRIMITIVE_TYPE_INTERNAL_ERROR,
            MATERIAL_TYPE_INTERNAL_ERROR,
            SURFACE_LOADER_INTERNAL_ERROR,
            TRANSFORM_TYPE_INTERNAL_ERROR,
            MEDIUM_TYPE_INTERNAL_ERROR,
            LIGHT_TYPE_INTERNAL_ERRROR,
            CAMERA_TYPE_INTERNAL_ERROR,
            // End
            END
        };

    private:
        Type            type;
        std::string     extra;

    public:
        // Constructors & Destructor
                    SceneError(Type);
                    SceneError(Type, const std::string&);
                    ~SceneError() = default;

        operator    Type() const;
        operator    std::string() const override;
};

class SceneException : public std::runtime_error
{
    private:
        SceneError          e;

    protected:
    public:
        SceneException(SceneError::Type t)
            : std::runtime_error("")
            , e(t)
        {}
        SceneException(SceneError::Type t, const std::string& s)
            : std::runtime_error("")
            , e(t, s)
        {}
        operator SceneError() const { return e; };
};

inline SceneError::SceneError(SceneError::Type t)
    : type(t)
{}

inline SceneError::SceneError(SceneError::Type t,
                              const std::string& ext)
    : type(t)
    , extra(ext)
{}

inline SceneError::operator Type() const
{
    return type;
}

inline SceneError::operator std::string() const
{
    static constexpr char const* const ErrorStrings[] =
    {
        "OK",
        // Common
        "Scene file not found",
        "Animation file not found",
        // Not Found
        "\"Materials\" array not found",
        "\"Primitives\" array not found",
        "\"Surfaces\" array not found",
        "\"Accelerators\" array not found",
        "\"Lights\" array not found",
        "\"Transforms\" array not found",
        "\"Cameras\" array not found",
        "\"Mediums\" array not found",
        "\"BaseAccelerator\" node not found",
        "\"BaseBoundaryLight\" node not found",
        "\"BaseMedium\" node not found",
        "\"LightSurfaces\" node not found",
        "\"CameraSurfaces\" node not found",
        // No Logic
        "No logic found for that accelerator",
        "No logic found for that material",
        "No logic found for that primitive",
        "No logic found for loading that surface data",
        "No logic found for that transform",
        "No logic found for that medium",
        "No logic found for that camera",
        "No logic found for that light",
        "No logic found for that tracer",
        // Id Errors
        "Duplicate accelerator id",
        "Duplicate material id",
        "Duplicate primitive id",
        "Duplicate transform id",
        "Duplicate medium id",
        "Duplicate light id",
        "Duplicate camera id",
        //
        "Accelerator id not found",
        "Material id not found",
        "Primitive id not found",
        "Transform id not found",
        "Medium id not found",
        "Light id not found",
        "Camera id not found",
        "Texture id not found",
        // Json Parse Errors
        "Logics does not match",
        "JSON type does not match with required type",
        "JSON file could not be parsed properly",
        // Special Type Values
        "Transform type name is unknown",
        "Light type name is unknown",
        // Loading Surface Data
        "Surface data type is mismatched with primitive type",
        "Surface data type not found",
        "Surface data unknown type",
        // Some Mat/Accel Logic
        // may not support certain prims
        "Primitive-Material mismatch",
        "Primitive-Accelerator mismatch",
        // Updating the scene
        // Primitive Update Size Mismatch
        "Updating primitive has more nodes than older itself",
        // Too many types than key system can handle
        "Accelerator groups required for this scene exceeds limit",
        "Accelerators in a group required for this scene exceeds limit",
        "Material groups required for this scene exceeds limit",
        "Materials in a batch required for this scene exceeds limit",
        // Texture Related
        "Texture access layout is unknown",
        "Texture type is unknown",
        "Unable to load texture file",
        "Texture Dimension does not match to the requested type",
        "Texture Channel does not match to the requested type",
        "Unable to find the texture file",
        "Cannot load a bitmap with multiple channels",
        // Misc
        "Too many data/material pairs per surface node",
        "Prim/Material pairs on surface node does not have same size",
        "Primitive types are not consistent in a surface",
        "Overlapping light found",
        "Overlapping camera found",
        "Primitive backed light cannot be a boundary light",
        // Internal Errors
        "Internal Error, Duplicate material id",
        "Internal Error, Duplicate accelerator id",
        //
        "Internal Error on the Primitive Type",
        "Internal Error on the Material Type",
        "Internal Error on the Surface Loader",
        "Internal Error on the Transform Type",
        "Internal Error on the Medium Type",
        "Internal Error on the Light Type",
        "Internal Error on the Camera Type"
    };
    static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(SceneError::END),
                  "Enum and enum string list size mismatch.");

    if(extra.empty())
        return ErrorStrings[static_cast<int>(type)];
    else
        return (std::string(ErrorStrings[static_cast<int>(type)])
                + " (" + extra + ')');
}