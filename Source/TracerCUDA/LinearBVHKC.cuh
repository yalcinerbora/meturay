#pragma once

#include "RayLib/AABB.h"

__device__ __forceinline__
AABB3f GenPointAABB(const Vector3f& point)
{
    return AABB3f(point, point);
}

__device__ __forceinline__
float DistancePoint(const Vector3f& point, const Vector3f& worldPoint)
{
    return (point - worldPoint).Length();
}