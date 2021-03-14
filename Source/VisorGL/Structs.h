#pragma once

struct ToneMapOptions
{
    bool    doToneMap;
    bool    doGamma;
    float   gamma;
};

static constexpr ToneMapOptions DefaultTMOptions =
{
    false,
    false,
    1.0f
};