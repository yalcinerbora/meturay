#pragma once

#include "Vector.h"
#include "AABB.h"

namespace Triangle
{
	template <class T>
	__device__ __host__
	AABB<3, T> BoundingBox(const Vector<3, T>& p0,
						   const Vector<3, T>& p1,
						   const Vector<3, T>& p2);
}

template <class T>
__device__ __host__
AABB<3, T> Triangle::BoundingBox(const Vector<3, T>& p0,
								 const Vector<3, T>& p1,
								 const Vector<3, T>& p2)
{
	AABB3f aabb(p0, p0);	
	aabb.SetMin(Vector3f::Min(aabb.Min(), p1));
	aabb.SetMin(Vector3f::Min(aabb.Min(), p2));

	aabb.SetMax(Vector3f::Max(aabb.Max(), p1));
	aabb.SetMax(Vector3f::Max(aabb.Max(), p2));
	return aabb;
}