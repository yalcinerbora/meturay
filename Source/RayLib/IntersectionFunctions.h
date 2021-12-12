#pragma once

#include "AABB.h"
#include "Vector.h"

namespace IntersectionFunctions
{
    __device__ __host__
    bool AABBIntersectsSphere(const AABB3f& abb,
                              const Vector3f& sphrPos,
                              float sphrRadius);
}

__device__ __host__ HYBRID_INLINE
bool IntersectionFunctions::AABBIntersectsSphere(const AABB3f& abb,
                                                 const Vector3f& sphrPos,
                                                 float sphrRadius)
{
    return false;
}