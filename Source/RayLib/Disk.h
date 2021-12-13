#pragma once

#include "Vector.h"
#include "AABB.h"
#include "Quaternion.h"

namespace Disk
{
    template <class T>
    __device__ __host__
    Vector3f SamplePoint(const Vector<3, T>& center,
                         const Vector<3, T>& normal,
                         T radius,
                         // 2 Uniform Samples
                         const Vector2f& xi);
}

template <class T>
__device__ __host__ HYBRID_INLINE
Vector3f Disk::SamplePoint(const Vector<3, T>& center,
                           const Vector<3, T>& normal,
                           T radius,
                           // 2 Uniform Samples
                           const Vector2f& xi)
{
    float r = xi * radius;
    float tetha = xi * 2.0f * MathConstants::Pi;

    // Aligned to Axis Z
    Vector3 diskPoint = Vector3(sqrt(r) * cos(tetha),
                                sqrt(r) * sin(tetha),
                                0.0f);

    // Rotate to disk normal
    QuatF rotation = Quat::RotationBetweenZAxis(normal);
    Vector3 rotatedPoint = rotation.ApplyRotation(diskPoint);
    Vector3 worldPoint = center + rotatedPoint;

    return worldPoint;
}