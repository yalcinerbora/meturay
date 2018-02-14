#pragma once

template<class T>
__device__ __host__
inline constexpr Quaternion<T>::Quaternion()
	: vec(1, 0, 0, 0)
{}

template<class T>
__device__ __host__
inline constexpr Quaternion<T>::Quaternion(float w, float x, float y, float z)
	: vec(w, x, y, z)
{}

template<class T>
__device__ __host__
inline constexpr Quaternion<T>::Quaternion(const float* v)
	: vec(v)
{}

template<class T>
__device__ __host__ 
Quaternion<T>::Quaternion(float angle, const Vector3& axis)
{
	
}

template<class T>
__device__ __host__	
inline Quaternion<T>::operator Vector4&()
{
	return vec;
}

template<class T>
__device__ __host__	
inline Quaternion<T>::operator const Vector4&() const
{
	return vec;
}

template<class T>
__device__ __host__ 
inline T& Quaternion<T>::operator[](int i)
{
	return vec[i];
}

template<class T>
__device__ __host__ 
inline const T& Quaternion<T>::operator[](int i) const
{
	return vec[i];
}
