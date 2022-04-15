#pragma once

template<int N, class T>
__device__ __host__ HYBRID_INLINE
constexpr AABB<N, T>::AABB(const Vector<N, T>& min,
                           const Vector<N, T>& max)
    : min(min)
    , max(max)
{}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
AABB<N, T>::AABB(const T* dataMin,
                        const T* dataMax)
    : min(dataMin)
    , max(dataMax)
{}

template<int N, class T>
template <class... Args0, class... Args1, typename, typename>
__device__ __host__ HYBRID_INLINE
constexpr AABB<N, T>::AABB(const Args0... dataList0,
                           const Args1... dataList1)
    : min(dataList0...)
    , max(dataList1...)
{}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
const Vector<N, T>& AABB<N, T>::Min() const
{
    return min;
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
const Vector<N, T>& AABB<N, T>::Max() const
{
    return max;
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> AABB<N, T>::Min()
{
    return min;
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> AABB<N, T>::Max()
{
    return max;
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
void AABB<N, T>::SetMin(const Vector<N, T>& v)
{
    min = v;
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
void AABB<N, T>::SetMax(const Vector<N, T>& v)
{
    max = v;
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> AABB<N, T>::Span() const
{
    return (max - min);
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> AABB<N, T>::Centroid() const
{
    return min + (Span() * static_cast<T>(0.5));
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
AABB<N, T> AABB<N, T>::Union(const AABB<N, T>& aabb) const
{
    return AABB<N, T>(Vector<N, T>::Min(min, aabb.min),
                      Vector<N, T>::Max(max, aabb.max));
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
AABB<N, T>& AABB<N, T>::UnionSelf(const AABB<N, T>& aabb)
{
    min = Vector<N, T>::Min(min, aabb.min);
    max = Vector<N, T>::Max(max, aabb.max);
    return *this;
}


template<int N, class T>
__device__ __host__ HYBRID_INLINE
bool AABB<N, T>::IsInside(const Vector<N, T>& point) const
{
    bool result = true;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        result &= (point[i] >= min[i] && point[i] <= max[i]);
    }
    return result;
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
bool AABB<N, T>::IsOutside(const Vector<N, T>& point) const
{
    return !IsInside(point);
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> AABB<N, T>::FurthestCorner(const Vector<N, T>& point) const
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    Vector<N, T> result;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        T minDist = abs(point[i] - min[i]);
        T maxDist = abs(point[i] - max[i]);
        result[i] = (minDist > maxDist) ? min[i] : max[i];
    }
    return result;
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
bool  AABB<N, T>::IntersectsSphere(const Vector3f& sphrPos,
                                   float sphrRadius)
{
    // Graphics Gems 2
    // http://www.realtimerendering.com/resources/GraphicsGems/gems/BoxSphere.c
    T dmin = 0;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        if(sphrPos[i] < min[i])
            dmin +=(sphrPos[i] - min[i]) * (sphrPos[i] - min[i]);
        else if(sphrPos[i] > max[i])
            dmin += (sphrPos[i] - max[i]) * (sphrPos[i] - max[i]);
    }
    if(dmin <= sphrRadius * sphrRadius)
        return true;
    return false;
}