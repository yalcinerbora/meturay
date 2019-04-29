#pragma once

#include "RayLib/Vector.h"

struct ConstantAlbedoMatData
{
    const Vector3* dAlbedo;
};

struct ConstantBoundaryMatData
{
    Vector3 backgroundColor;
};