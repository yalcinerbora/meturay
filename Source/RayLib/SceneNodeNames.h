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
    static constexpr const char* BASE_MEDIUM = "BaseMedium";
    static constexpr const char* LIGHT_SURFACE_BASE = "LightSurfaces";
    static constexpr const char* CAMERA_SURFACE_BASE = "CameraSurfaces";

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
    // Material & Light Common Names
    static constexpr const char* MEDIUM = "medium";
    static constexpr const char* LIGHT = "light";
    // Light Related Names
    // Light Type Values
    static constexpr const char* LIGHT_POSITION = POSITION;
    static constexpr const char* LIGHT_DIRECTION = "direction";
    static constexpr const char* LIGHT_SPHR_CENTER = "center";
    static constexpr const char* LIGHT_DISK_CENTER = LIGHT_SPHR_CENTER;
    static constexpr const char* LIGHT_SPHR_RADIUS = "radius";
    static constexpr const char* LIGHT_DISK_RADIUS = LIGHT_SPHR_RADIUS;
    static constexpr const char* LIGHT_RECT_V0 = "v0";
    static constexpr const char* LIGHT_RECT_V1 = "v1";
    static constexpr const char* LIGHT_CONE_APERTURE = "aperture";
    static constexpr const char* LIGHT_MATERIAL = MATERIAL;
    static constexpr const char* LIGHT_PRIMITIVE = PRIMITIVE;
    // Texture Related Names
    static constexpr const char* TEXTURE_IS_CACHED = "isCached";
    static constexpr const char* TEXTURE_FILTER = "filter";
    // Indentity Transform Type Name
    static constexpr const char* TRANSFORM_IDENTITY = "Identity";


}