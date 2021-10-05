#pragma once

#include <cstdint>

#include "RayLib/Vector.h"

template<class T>
struct ImageGMem
{
    T*          gPixels;
    uint32_t*   gSampleCounts;
};

template<class T>
struct ImageGMemConst
{
    const T* gPixels;
    const uint32_t* gSampleCounts;
};