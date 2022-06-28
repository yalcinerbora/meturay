#pragma once

#include <cstdint>

#include "RayLib/Vector.h"

template<class T>
struct ImageGMem
{
    T*          gPixels;
    float*      gSampleCounts;
};

template<class T>
struct ImageGMemConst
{
    const T*        gPixels;
    const float*    gSampleCounts;
};