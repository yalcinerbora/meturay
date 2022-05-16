#pragma once

#include <array>
#include <string>
#include  "RayLib/TracerError.h"

enum class WFPGRenderMode
{
    // Tracer outputs camera radiance values as normal
    NORMAL,

    // Tracer traces camera rays over the SVO,
    // outputs false color for each leaf.
    SVO_FALSE_COLOR,

    // Tracer traces camera rays over the scene
    // using the classic accelerators,
    // then queries the hit positions over the SVO
    // and outputs false color for each leaf.
    SVO_INITIAL_HIT_QUERY,

    // Tracer does path tracing (with path guiding)
    // normally.
    // However; after all bounces are calculated and SVO
    // radiance is updated, it does one last SVO trace to query
    // radiances and outputs that values to the framebuffer instead
    SVO_RADIANCE,
    //
    END
};

enum class WFPGFilterMode
{
    NEAREST,
    BOX,
    TENT,

    END
};

static constexpr std::array<std::string_view, static_cast<size_t>(WFPGRenderMode::END)> WFPGRenderModeNames =
{
    "Normal"
    "SVOFalseColor",
    "SVOInitialHit",
    "SVORadiance"
};

static constexpr std::array<std::string_view, static_cast<size_t>(WFPGFilterMode::END)> WFPGFilterModeNames =
{
    "Nearest",
    "Box",
    "Tent"
};

static TracerError StringToWFPGRenderMode(WFPGRenderMode& m, const std::string& s)
{
    for(int i = 0; i < static_cast<int>(WFPGRenderMode::END); i++)
    {
        if(s == WFPGRenderModeNames[i])
        {
            m = static_cast<WFPGRenderMode>(i);
            return TracerError::OK;
        }
    }
    return TracerError::TRACER_INTERNAL_ERROR;
}

static std::string WFPGRenderModeToString(WFPGRenderMode m)
{
    return std::string(WFPGRenderModeNames[static_cast<int>(m)]);
}

static TracerError StringToWFPGFilterMode(WFPGFilterMode& m, const std::string& s)
{
    for(int i = 0; i < static_cast<int>(WFPGFilterMode::END); i++)
    {
        if(s == WFPGFilterModeNames[i])
        {
            m = static_cast<WFPGFilterMode>(i);
            return TracerError::OK;
        }
    }
    return TracerError::TRACER_INTERNAL_ERROR;
}

static std::string WFPGFilterModeToString(WFPGFilterMode m)
{
    return std::string(WFPGFilterModeNames[static_cast<int>(m)]);
}