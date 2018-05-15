#pragma once

#include "Constans.h"

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
	return Quaternion(vec[0] * right[0] - vec[1] * right[1] - vec[2] * right[2] - vec[3] * right[3],		// W
					  vec[0] * right[1] + vec[1] * right[0] + vec[2] * right[3] - vec[3] * right[2],		// X
					  vec[0] * right[2] + vec[2] * right[0] + vec[3] * right[1] - vec[1] * right[3],		// Y
					  vec[0] * right[3] + vec[3] * right[0] + vec[1] * right[2] - vec[2] * right[1]);		// Z
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator*(float right) const
{
	return Quaternion(vec * t);
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator+(const Quaternion& right) const
{
	return Quaternion(vec + right.vec);
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator-(const Quaternion& right) const
{
	return Quaternion(vec - right.vec);
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator-() const
{
	return Quaternion(-vec);
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::operator/(float right) const
{
	return Quaternion(vec / right);
}

template<class T>
__device__ __host__
inline void Quaternion<T>::operator*=(const Quaternion& right)
{
	Quaternion copy(*this);
	(*this) = copy * right;
}

template<class T>
__device__ __host__ 
inline void Quaternion<T>::operator*=(float right)
{
	vec *= right;
}

template<class T>
__device__ __host__ 
inline void Quaternion<T>::operator+=(const Quaternion& right)
{
	vec += right;
}

template<class T>
__device__ __host__ 
inline void Quaternion<T>::operator-=(const Quaternion& right)
{
	vec -= right;
}

template<class T>
__device__ __host__ 
inline void Quaternion<T>::operator/=(float right)
{
	vec /= right;
}

template<class T>
__device__ __host__ 
inline bool Quaternion<T>::operator==(const Quaternion& right)
{
	return vec == right.vec;
}

template<class T>
__device__ __host__ 
inline bool Quaternion<T>::operator!=(const Quaternion& right)
{
	return vec != right.vec;
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::Normalize() const
{
	return Quaternion(vec.Normalize());
}

template<class T>
__device__ __host__ 
inline Quaternion<T>& Quaternion<T>::NormalizeSelf()
{
	vec.NormalizeSelf();
}

template<class T>
__device__ __host__ 
inline T Quaternion<T>::Length() const
{
	return vec.Length();
}

template<class T>
__device__ __host__ 
inline T Quaternion<T>::LengthSqr() const
{
	return vec.LengthSqr();
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::Conjugate() const
{
	return Quaternion(vec[0], -vec[1], -vec[2], -vec[3]);
}

template<class T>
__device__ __host__ 
inline Quaternion<T>& Quaternion<T>::ConjugateSelf()
{
	vec[1] = -vec[1];
	vec[2] = -vec[2];
	vec[3] = -vec[3];
	return *this;
}

template<class T>
__device__ __host__ 
inline T Quaternion<T>::DotProduct(const Quaternion& right) const
{
	vec.Dot(right.vec);
}

template<class T>
__device__ __host__ 
inline Vector<3, T> Quaternion<T>::ApplyRotation(const Vector<3, T>& vector) const
{
	// q * v * qInv
	// .Normalize();
	Quaternion qInv = Conjugate();
	Quaternion vectorQ(0.0f, vector[0], vector[1], vector[2]);

	Quaternion result((*this) * (vectorQ * qInv));
	return Vector<3,T>(result[0], result[1], result[2]);
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::NLerp(const Quaternion& start, const Quaternion& end, float t)
{
	return (start + t * (end - start));// .Normalize();
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::SLerp(const Quaternion& start, const Quaternion& end, float t)
{
	float cosTetha = start.DotProduct(end);
	// SLerp
	float angle = std::acos(cosTetha);
	return (start * std::sin(angle * (1.0f - t)) + end * std::sin(angle * t)) 
		/ std::sin(angle);
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::RotationBetween(const Vector<3,T>& a, const Vector<3,T>& b)
{
	Vector3 aCrossB = a.Cross(b);
	float aDotB = a.Dot(b);
	if(aCrossB != Zero3)
		aCrossB.NormalizeSelf();
	return Quaternion(std::acos(aDotB), aCrossB);
}

template<class T>
__device__ __host__ 
inline Quaternion<T> Quaternion<T>::RotationBetweenZAxis(const Vector<3, T>& b)
{
	Vector<3, T> zCrossD(-b[1], b[0], 0.0f);
	float zDotD = b[2];

	// Half angle teorem
	float sin = std::sqrt((1.0f - zDotD) * 0.5f);
	float cos = std::sqrt((zDotD + 1.0f) * 0.5f);

	zCrossD.NormalizeSelf();
	float x = zCrossD[0] * sin;
	float y = zCrossD[1] * sin;
	float z = zCrossD[2] * sin;
	float w = cos;
	if(std::abs(zDotD + 1.0f) < MathConstants::Epsilon)
	{
		return Quaternion(MathConstants::PI, Vector<3, T>::YAxis);
	}
	return Quaternion(w, x, y, z);
}