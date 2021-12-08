#pragma once

namespace NodeNames
{
    static constexpr const char* SCENE_EXT = "mscene";
    static constexpr const char* ANIM_EXT = "manim";
    // Common Base Arrays
    static constexpr const char* CAMERA_BASE = "Cameras";
    static constexpr const char* LIGHT_BASE = "Lights";
    static constexpr const char* MEDIUM_BASE = "Mediums";
    static constexpr const char* TEXTURE_BASE = "Textures";
    static constexpr const char* ACCELERATOR_BASE = "Accelerators";
    static constexpr const char* TRANSFORM_BASE = "Transforms";
    static constexpr const char* PRIMITIVE_BASE = "Primitives";
    static constexpr const char* MATERIAL_BASE = "Materials";
    static constexpr const char* SURFACE_BASE = "Surfaces";
    static constexpr const char* SURFACE_DATA_BASE = "SurfaceData";
    static constexpr const char* BASE_ACCELERATOR = "BaseAccelerator";
    static constexpr const char* BASE_BOUNDARY_LIGHT = "BaseBoundaryLight";
    static constexpr const char* BASE_MEDIUM = "BaseMedium";
    static constexpr const char* BASE_BOUNDARY_TRANSFORM = "BaseBoundaryTransform";

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
    static constexpr const char* CAMERA = "camera";
    // Texture Related Names
    static constexpr const char* TEXTURE_IS_CACHED = "isCached";
    static constexpr const char* TEXTURE_FILTER = "filter";
    static constexpr const char* TEXTURE_FILE = "file";
    static constexpr const char* TEXTURE_SIGNED = "signed";
    static constexpr const char* TEXTURE_CHANNEL = "channels";
    static constexpr const char* TEXTURE_NAME = "texture";
    // Light Related
    static constexpr const char* LIGHT_TYPE_PRIMITIVE = "Primitive";

    // Identity Transform Type Name
    static constexpr const char* TRANSFORM_IDENTITY = "Identity";
}