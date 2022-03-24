#pragma once

#include "Vector.h"
#include "AABB.h"
#include "Quaternion.h"
#include "Matrix.h"

#include "Log.h"

namespace Triangle
{
    template <class T>
    __device__ __host__ HYBRID_INLINE
    AABB<3, T> BoundingBox(const Vector<3, T>& p0,
                           const Vector<3, T>& p1,
                           const Vector<3, T>& p2);

    template <class T>
    __device__ __host__ HYBRID_INLINE
    Vector<3, T> CalculateTangent(const Vector<3, T>& p0,
                                  const Vector<3, T>& p1,
                                  const Vector<3, T>& p2,

                                  const Vector<2, T>& uv0,
                                  const Vector<2, T>& uv1,
                                  const Vector<2, T>& uv2);

    template <class T>
    __device__ __host__ HYBRID_INLINE
    void LocalRotation(Quaternion<T>&,
                       Quaternion<T>&,
                       Quaternion<T>&,
                       const Vector<3, T>* positions,
                       const Vector<3, T>* normals,
                       const Vector<2, T>* uvs);

    template <class T>
    __device__ __host__ HYBRID_INLINE
    void LocalRotation(Quaternion<T>&,
                       Quaternion<T>&,
                       Quaternion<T>&,
                       const Vector<3, T>* normals,
                       const Vector<3, T>* tangents);

    template <class T>
    __device__ __host__ HYBRID_INLINE
    T Area(const Vector<3, T>* positions);

    template <class T>
    __device__ __host__ HYBRID_INLINE
    Vector<3, T> Normal(const Vector<3, T>* positions);
}

template <class T>
__device__ __host__
AABB<3, T> Triangle::BoundingBox(const Vector<3, T>& p0,
                                 const Vector<3, T>& p1,
                                 const Vector<3, T>& p2)
{
    AABB<3, T> aabb(p0, p0);
    aabb.SetMin(Vector<3, T>::Min(aabb.Min(), p1));
    aabb.SetMin(Vector<3, T>::Min(aabb.Min(), p2));

    aabb.SetMax(Vector<3, T>::Max(aabb.Max(), p1));
    aabb.SetMax(Vector<3, T>::Max(aabb.Max(), p2));
    return aabb;
}

template <class T>
__device__ __host__
Vector<3, T> Triangle::CalculateTangent(const Vector<3, T>& p0,
                                        const Vector<3, T>& p1,
                                        const Vector<3, T>& p2,

                                        const Vector<2, T>& uv0,
                                        const Vector<2, T>& uv1,
                                        const Vector<2, T>& uv2)
{
    // Edges (Tri is CCW)
    Vector<3, T> vec0 = p1 - p0;
    Vector<3, T> vec1 = p2 - p0;

    Vector<2, T> dUV0 = uv1 - uv0;
    Vector<2, T> dUV1 = uv2 - uv0;

    T t = (dUV0[0] * dUV1[1] -
           dUV1[0] * dUV0[1]);

    Vector<3, T> tangent;
    tangent = t * (dUV1[1] * vec0 - dUV0[1] * vec1);
    tangent.NormalizeSelf();
    return tangent;
}

template <class T>
__device__ __host__
void Triangle::LocalRotation(Quaternion<T>& q0,
                             Quaternion<T>& q1,
                             Quaternion<T>& q2,

                             const Vector<3, T>* n,
                             const Vector<3, T>* t)
{
    Vector<3, T> b0 = Cross(n[0], t[0]);
    Vector<3, T> b1 = Cross(n[1], t[1]);
    Vector<3, T> b2 = Cross(n[2], t[2]);

    TransformGen::Space(q0, t[0], b0, n[0]);
    TransformGen::Space(q1, t[1], b1, n[1]);
    TransformGen::Space(q2, t[2], b2, n[2]);

    q0.ConjugateSelf();
    q1.ConjugateSelf();
    q2.ConjugateSelf();
}

template <class T>
__device__ __host__
void Triangle::LocalRotation(Quaternion<T>& q0,
                             Quaternion<T>& q1,
                             Quaternion<T>& q2,

                             const Vector<3, T>* p,
                             const Vector<3, T>* n,
                             const Vector<2, T>* uv)
{
    // We calculate tangent once
    // is this consistent? (should i calculate for all vertices of tri?
    Vector<3, T> t0 = CalculateTangent<T>(p[0], p[1], p[2], uv[0], uv[1], uv[2]);
    Vector<3, T> t1 = CalculateTangent<T>(p[1], p[2], p[0], uv[1], uv[2], uv[0]);
    Vector<3, T> t2 = CalculateTangent<T>(p[2], p[0], p[1], uv[2], uv[0], uv[1]);
    //Vector<3, T> t1 = t0;
    //Vector<3, T> t2 = t0;

    // Degenerate triangle is found,
    // (or uv's are degenerate)
    // arbitrarily find a tangent
    if(t0.HasNaN()) t0 = OrthogonalVector(n[0]);
    if(t1.HasNaN()) t1 = OrthogonalVector(n[1]);
    if(t2.HasNaN()) t2 = OrthogonalVector(n[2]);

    // Gram-Schmidt orthonormalization
    // This is required since normal may be skewed to hide
    // edges (to create smooth lighting)
    t0 = (t0 - n[0] * n[0].Dot(t0)).Normalize();
    t1 = (t1 - n[1] * n[1].Dot(t1)).Normalize();
    t2 = (t2 - n[2] * n[2].Dot(t2)).Normalize();

    Vector<3, T> b0 = Cross(n[0], t0);
    Vector<3, T> b1 = Cross(n[1], t1);
    Vector<3, T> b2 = Cross(n[2], t2);

    //printf("T %f, %f, %f - B %f, %f, %f - N %f, %f, %f\n"
    //       "T %f, %f, %f - B %f, %f, %f - N %f, %f, %f\n"
    //       "T %f, %f, %f - B %f, %f, %f - N %f, %f, %f\n"
    //       "====================================\n",
    //       t0[0], t0[1], t0[2],
    //       b0[0], b0[1], b0[2],
    //       n[0][0], n[0][1], n[0][2],
    //
    //       t1[0], t1[1], t1[2],
    //       b1[0], b1[1], b1[2],
    //       n[1][0], n[1][1], n[1][2],
    //
    //       t2[0], t2[1], t2[2],
    //       b2[0], b2[1], b2[2],
    //       n[2][0], n[2][1], n[2][2]);

    TransformGen::Space(q0, t0, b0, n[0]);
    TransformGen::Space(q1, t1, b1, n[1]);
    TransformGen::Space(q2, t2, b2, n[2]);
}

template <class T>
__device__ __host__ HYBRID_INLINE
T Triangle::Area(const Vector<3, T>* positions)
{
    Vector<3, T> e0 = positions[1] - positions[0];
    Vector<3, T> e1 = positions[2] - positions[0];

    return Cross(e0, e1).Length() * static_cast<T>(0.5);
}

template <class T>
__device__ __host__ HYBRID_INLINE
Vector<3, T> Triangle::Normal(const Vector<3, T>* positions)
{
    Vector<3, T> e0 = positions[1] - positions[0];
    Vector<3, T> e1 = positions[2] - positions[0];

    return  Cross(e0, e1).Normalize();
}