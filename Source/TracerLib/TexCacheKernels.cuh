#pragma once

// List of functions that are required by the GPU texture cache

// GPU texture cache is responsible for converting
// UV lists (determined by the user) to acual UV lists that hold
// cached texture coordinates

// In basic implementation
// it will support 2D textures only with 16-bit uv's and single layered textures
// 

#include "RayLib/Vector.h"


struct BasicUV
{
    Vector2us   unormUV;
    uint32_t    textureId;
};

struct BasicUVList
{
    BasicUV* gList;
};


//
template<class UVList, class MGroup, class PGroup>
using OutputUVsFunc = void(*)(// Output
                              UVList&,
                              // Input
                              HitStructs,
                              primitiveId,
                              MGroup::Surface& surface);


//template<class TexCache, class MGroup>
//__global__
//void KC()