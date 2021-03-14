#pragma once

struct ToneMapOptions
{
    bool    doGamma;
    bool    doToneMap;
    float   gamma;
};

static constexpr ToneMapOptions DefaultTMOptions =
{
    false,
    false,
    1.0f
};