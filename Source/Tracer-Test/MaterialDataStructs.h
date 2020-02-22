#pragma once

#include "RayLib/Vector.h"

struct NullData {};

struct ConstantAlbedoMatData
{
    const Vector3* dAlbedo;
};

struct ConstantMediumMatData
{
    const Vector4* dAlbedoAndIndex;
    
};

//struct ConstantBoundaryMatData
//{
//    Vector3 backgroundColor;
//};