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
    min = Vector<N, T>::Min(min, aabb.min),
        max = Vector<N, T>::Max(max, aabb.max);
    return *this;
}


template<int N, class T>
__device__ __host__ HYBRID_INLINE
bool AABB<N, T>::IsInside(const Vector<N, T>& point)
{
    bool result = true;
    result &= (point[0] >= min[0] && point[0] <= max[0]);
    result &= (point[1] >= min[1] && point[1] <= max[1]);
    result &= (point[2] >= min[2] && point[2] <= max[2]);
    return result;
}

template<int N, class T>
__device__ __host__ HYBRID_INLINE
bool AABB<N, T>::IsOutside(const Vector<N, T>& point)
{
    return !IsInside(point);
}