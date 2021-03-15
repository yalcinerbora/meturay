#pragma once

struct ToneMapOptions
{
    bool    doToneMap;
    bool    doGamma;
    float   gamma;
    float   burnRatio;
};

static constexpr ToneMapOptions DefaultTMOptions =
{
    false,
    false,
    1.0f,
    1.0f
};