#pragma once

struct ToneMapOptions
{
    bool    doToneMap;
    bool    doGamma;
    bool    doKeyAdjust;
    float   gamma;
    float   burnRatio;
    float   key;
};

static constexpr ToneMapOptions DefaultTMOptions =
{
    true,      // Do tone-map
    true,      // Do gamma
    false,     // Do key adjust
    2.2f,      // sRGB Gamma
    1.0f,      // Utilize Lwhite as is
    0.18f      // Reinhard 2002 Key value
};