#pragma once

#include <cstdint>

#include "RayLib/Vector.h"

template<class T>
struct ImageGMem
{
    T* gPixels;
    uint32_t* gSampleCount;
};

template<class T>
struct ImageGMemConst
{
    const T* gPixels;
    const uint32_t* gSampleCount;
};