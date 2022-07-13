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

template <class T>
struct CamSampleGMem
{
    T*        gValues;
    Vector2f* gImgCoords;
};

template <class T>
struct CamSampleGMemConst
{
    const T*        gValues;
    const Vector2f* gImgCoords;
};