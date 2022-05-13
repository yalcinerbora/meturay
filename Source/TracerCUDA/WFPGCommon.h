#pragma once

#include <array>
#include <string>
#include  "RayLib/TracerError.h"

enum class VoxelTraceMode
{
    FALSE_COLOR,
    RADIANCE,

    END
};


enum class RadianceFilterMode
{
    BOX,
    TENT,
};

static constexpr std::array<std::string_view, static_cast<size_t>(VoxelTraceMode::END)> VoxelTraceModeNames =
{
    "FalseColor",
    "Radiance"
};

static TracerError StringToVoxelTraceMode(VoxelTraceMode& m, const std::string& s)
{
    for(int i = 0; i < static_cast<int>(VoxelTraceMode::END); i++)
    {
        if(s == VoxelTraceModeNames[i])
        {
            m = static_cast<VoxelTraceMode>(i);
            return TracerError::OK;
        }
    }
    return TracerError::TRACER_INTERNAL_ERROR;
}

static std::string VoxelTraceModeToString(VoxelTraceMode m)
{
    return std::string(VoxelTraceModeNames[static_cast<int>(m)]);
}