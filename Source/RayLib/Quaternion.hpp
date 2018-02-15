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
inline Quaternion<T>::Quaternion(float angle, const Vector3& axis)
{
	angle *= 0.5;
	T sinAngle = std::sin(angle);

	vec[1] = axis[0] * sinAngle;
	vec[2] = axis[1] * sinAngle;
	vec[3] = axis[2] * sinAngle;
	vec[0] = std::cos(angle);
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
inline Quaternion<T>::operator float*()
{
	return static_cast<float*>(vec);
}

template<class T>
__device__ __host__	
inline Quaternion<T>::operator const float*() const
{
	return static_cast<const float*>(vec);
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

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator*(const Quaternion& right) const
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator*(float right) const
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator+(const Quaternion& right) const
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator-(const Quaternion& right) const
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator-() const
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator/(float right) const
{

}

template<class T>
__device__ __host__
inline void Quaternion<T>::operator*=(const Quaternion& right)
{

}

template<class T>
__device__ __host__ 
inline void Quaternion<T>::operator*=(float right)
{

}

template<class T>
__device__ __host__ 
inline void Quaternion<T>::operator+=(const Quaternion& right)
{

}

template<class T>
__device__ __host__ 
inline void Quaternion<T>::operator-=(const Quaternion& right)
{

}

template<class T>
__device__ __host__ 
inline void Quaternion<T>::operator/=(float right)
{

}

template<class T>
__device__ __host__ 
inline bool Quaternion<T>::operator==(const Quaternion& right)
{

}

template<class T>
__device__ __host__ 
inline bool Quaternion<T>::operator!=(const Quaternion& right)
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::Normalize() const
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T>& Quaternion<T>::NormalizeSelf()
{

}

template<class T>
__device__ __host__ 
inline T Quaternion<T>::Length() const
{

}

template<class T>
__device__ __host__ 
inline T Quaternion<T>::LengthSqr() const
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::Conjugate() const
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T>& Quaternion<T>::ConjugateSelf()
{

}

template<class T>
__device__ __host__ 
inline T Quaternion<T>::DotProduct(const Quaternion&) const
{

}

template<class T>
__device__ __host__ 
inline Vector<3, T> Quaternion<T>::ApplyRotation(const Vector<3, T>& vec) const
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::NLerp(const Quaternion& start, const Quaternion& end, float t)
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::SLerp(const Quaternion& start, const Quaternion& end, float t)
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::RotationBetween(const Vector<3,T>& a, const Vector<3,T>& b)
{

}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::RotationBetweenZAxis(const Vector<3, T>& b)
{

}