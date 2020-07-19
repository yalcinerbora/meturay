#pragma once

namespace NodeNames
{
    static constexpr const char* SCENE_EXT = "mscene";
    static constexpr const char* ANIM_EXT = "manim";
    // Common Base Arrays
    static constexpr const char* CAMERA_BASE = "Cameras";
    static constexpr const char* LIGHT_BASE = "Lights";
    static constexpr const char* MEDIUM_BASE = "Mediums";
    static constexpr const char* ACCELERATOR_BASE = "Accelerators";
    static constexpr const char* TRANSFORM_BASE = "Transforms";
    static constexpr const char* PRIMITIVE_BASE = "Primitives";
    static constexpr const char* MATERIAL_BASE = "Materials";
    static constexpr const char* SURFACE_BASE = "Surfaces";
    static constexpr const char* SURFACE_DATA_BASE = "SurfaceData";
    static constexpr const char* BASE_ACCELERATOR = "BaseAccelerator";
    static constexpr const char* BASE_OUTSIDE_MATERIAL = "BaseBoundaryMaterial";
    static constexpr const char* TRACER_LOGIC = "Tracer";
    static constexpr const char* ESTIMATOR = "Estimator";
    // Common Names
    static constexpr const char* ID = "id";
    static constexpr const char* TYPE = "type";
    static constexpr const char* NAME = "name";
    static constexpr const char* TAG = "tag";
    // Common Names
    static constexpr const char* POSITION = "position";
    static constexpr const char* DATA = "data";
    // Surface Related Names
    static constexpr const char* TRANSFORM = "transform";
    static constexpr const char* PRIMITIVE = "primitive";
    static constexpr const char* ACCELERATOR = "accelerator";
    static constexpr const char* MATERIAL = "material";
    // Camera Related Names
    static constexpr const char* CAMERA_APERTURE = "apertureSize";
    static constexpr const char* CAMERA_FOCUS = "focusDistance";
    static constexpr const char* CAMERA_PLANES = "planes";
    static constexpr const char* CAMERA_FOV = "fov";
    static constexpr const char* CAMERA_GAZE = "gaze";
    static constexpr const char* CAMERA_UP = "up";
    // Medium Related Names
    static constexpr const char* MEDIUM_ABSORBTION = "absorption";
    static constexpr const char* MEDIUM_SCATTERING = "scattering";
    static constexpr const char* MEDIUM_IOR = "ior";
    static constexpr const char* MEDIUM_Phase = "phase";
    // Light Related Names
    // Light Type Values
    static constexpr const char* LIGHT_POWER = "power";
    static constexpr const char* LIGHT_POSITION = POSITION;
    static constexpr const char* LIGHT_DIRECTION = "direction";
    static constexpr const char* LIGHT_SPHR_CENTER = "center";
    static constexpr const char* LIGHT_DISK_CENTER = LIGHT_SPHR_CENTER;
    static constexpr const char* LIGHT_SPHR_RADIUS = "radius";
    static constexpr const char* LIGHT_DISK_RADIUS = LIGHT_SPHR_RADIUS;
    static constexpr const char* LIGHT_RECT_V0 = "v0";
    static constexpr const char* LIGHT_RECT_V1 = "v1";
    static constexpr const char* LIGHT_CONE_APERTURE = "aperture";
    // Texture Related Names
    static constexpr const char* TEXTURE_IS_CACHED = "isCached";
    static constexpr const char* TEXTURE_FILTER = "filter";
    // Common
    static constexpr const char* LIGHT_MATERIAL = "material";
    static constexpr const char* LIGHT_PRIMITIVE = "primitive";
    // Transform Related Names
    // Common
    static constexpr const char* TRANSFORM_FORM = "form";
    // Transform Form Values
    static constexpr const char* TRANSFORM_FORM_MATRIX4 = "matrix4x4";
    static constexpr const char* TRANSFORM_FORM_T_R_S = "transformRotateScale";
}