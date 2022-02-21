#pragma once

#include "RayLib/AABB.h"

struct PointStruct
{
    Vector3f position;
};

__device__ __host__ HYBRID_INLINE
AABB3f GenPointAABB(const PointStruct& point)
{
    return AABB3f(point.position, point.position);
}

class PointDistanceFunctor
{
    public:
        __device__ float    operator()(const PointStruct& leaf,
                                       const PointStruct& point) const;
};

__device__ inline
float PointDistanceFunctor::operator()(const PointStruct& leaf,
                                       const PointStruct& point) const
{
    return (point.position - leaf.position).Length();
}