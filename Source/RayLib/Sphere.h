#pragma once

#include "Vector.h"
#include "AABB.h"

namespace Sphere
{
    template<class T>
    __host__ __device__
    AABB<3, T> BoundingBox(const Vector<3, T>& center, T radius);
}

template<class T>
__host__ __device__
inline AABB<3, T> Sphere::BoundingBox(const Vector<3, T>& center, T radius)
{
    return AABB<3, T>(center - radius, center + radius);
}