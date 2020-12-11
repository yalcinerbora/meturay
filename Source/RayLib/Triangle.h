#pragma once

#include "Vector.h"
#include "AABB.h"
#include "Quaternion.h"
#include "Matrix.h"

namespace Triangle
{
    template <class T>
    __device__ __host__
    AABB<3, T> BoundingBox(const Vector<3, T>& p0,
                           const Vector<3, T>& p1,
                           const Vector<3, T>& p2);

    template <class T>
    __device__ __host__
    Vector<3, T> CalculateTangent(const Vector<3, T>& p0,
                                  const Vector<3, T>& p1,
                                  const Vector<3, T>& p2,

                                  const Vector<2, T>& uv0,
                                  const Vector<2, T>& uv1,
                                  const Vector<2, T>& uv2);

    template <class T>
    __device__ __host__
    void LocalRotation(Quaternion<T>&,
                       Quaternion<T>&,
                       Quaternion<T>&,
                       const Vector<3, T>* positions,
                       const Vector<3, T>* normals,
                       const Vector<2, T>* uvs);

    template <class T>
    __device__ __host__
    void LocalRotation(Quaternion<T>&,
                       Quaternion<T>&,
                       Quaternion<T>&,
                       const Vector<3, T>* normals,
                       const Vector<3, T>* tangents);

}

template <class T>
__device__ __host__
inline static AABB<3, T> Triangle::BoundingBox(const Vector<3, T>& p0,
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

#include "Log.h"

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

    Matrix<4, T> localToTangent(vec0[0], vec1[0], p0[0], 0,
                                vec0[1], vec1[1], p0[1], 0,
                                vec0[2], vec1[2], p0[2], 0,
                                0,       0,       0,     1);
    Matrix<4, T> localToTexture(dUV0[0], dUV1[0], uv0[0], 0,
                                dUV0[1], dUV1[1], uv0[1], 0,
                                0,       0,       1,      0,
                                0,       0,       0,      1);

    Matrix4x4 tbn = localToTangent.Transpose() * localToTexture.Transpose().Inverse();
    Vector3 Mtangent = tbn * XAxis;
    Vector3 Mbtangent = tbn * YAxis;
    Vector3 Mnormal = tbn * ZAxis;

    T t = 1 / (dUV0[0] * dUV1[1] -
               dUV1[0] * dUV0[1]);

    if(t != t)
    {
        METU_ERROR_LOG("NAN FOUND");
    }

    Vector<3, T> tangent;
    tangent = t * (dUV1[1] * vec0 - dUV0[1] * vec1);

    if(tangent[0] != tangent[0] ||
       tangent[1] != tangent[1] ||
       tangent[2] != tangent[2])
    {
        METU_ERROR_LOG("NAN FOUND");
    }

    Vector<3, T> tgntNormal = tangent.NormalizeSelf();

    if(tangent[0] != tangent[0] ||
       tangent[1] != tangent[1] ||
       tangent[2] != tangent[2])
    {
        METU_ERROR_LOG("NAN FOUND");
    }

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

    TransformGen::Space(q0, b0, t[0], n[0]);
    TransformGen::Space(q1, b1, t[1], n[1]);
    TransformGen::Space(q2, b2, t[2], n[2]);
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
    //Vector<3, T> t1 = CalculateTangent(p[1], p[2], p[0], uv[1], uv[2], uv[0]);
    //Vector<3, T> t2 = CalculateTangent(p[2], p[0], p[1], uv[2], uv[0], uv[1]);
    Vector<3, T> t1 = t0;
    Vector<3, T> t2 = t0;

    if(t0[0] != t0[0] || t0[1] != t0[1] || t0[2] != t0[2])
    {
        METU_ERROR_LOG("NAN FOUND");
    }

    // Gram–Schmidt othonormalization
    // This is required since normal may be skewed to hide
    // edges (to create smooth lighting)
    t0 = (t0 - n[0] * n[0].Dot(t0)).Normalize();
    t1 = (t1 - n[1] * n[1].Dot(t1)).Normalize();
    t2 = (t2 - n[2] * n[2].Dot(t2)).Normalize();

    Vector<3, T> b0 = Cross(t0, n[0]);
    Vector<3, T> b1 = Cross(t1, n[1]);
    Vector<3, T> b2 = Cross(t2, n[2]);

    if(t0[0] != t0[0] || t0[1] != t0[1] || t0[2] != t0[2] ||
       b0[0] != b0[0] || b0[1] != b0[1] || b0[2] != b0[2] ||
       n[0][0] != n[0][0] || n[0][1] != n[0][1] || n[0][2] != n[0][2])
    {
        METU_ERROR_LOG("NAN FOUND");
    }

    TransformGen::Space(q0, b0, t0, n[0]);
    TransformGen::Space(q1, b1, t1, n[1]);
    TransformGen::Space(q2, b2, t2, n[2]);
}