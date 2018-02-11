#pragma once

template <int N, class T>
__device__ __host__
inline constexpr Vector<N, T>::Vector()
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		vector[i] = 0.0f;
	}
}

template <int N, class T>
__device__ __host__
inline constexpr Vector<N, T>::Vector(float data)
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		vector[i] = data;
	}
}

template <int N, class T>
__device__ __host__
inline constexpr Vector<N, T>::Vector(const float* data)
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		vector[i] = data[i];
	}
}

template <int N, class T>
template <class... Args, typename>
__device__ __host__
inline constexpr Vector<N, T>::Vector(const Args... dataList)
	: vector{static_cast<T>(dataList) ...}
{
	static_assert(sizeof...(dataList) == N, "Vector constructor should have exact "
											"same count of template count "
											"as arguments");
}

template <int N, class T>
template <class... Args, typename>
__device__ __host__
inline constexpr Vector<N, T>::Vector(const Vector<N - sizeof...(Args), T>& v,
									  const Args... dataList)
{
	constexpr int vectorSize = N - sizeof...(dataList);
	static_assert(sizeof...(dataList) + vectorSize == N, "Total type count of the partial vector"
														 "constructor should exactly match vector size");

	UNROLL_LOOP
	for(int i = 0; i < vectorSize; i++)
	{
		vector[i] = v[i];
	}
	const T arr[] = {dataList...};
	UNROLL_LOOP
	for(int i = vectorSize; i < N; i++)
	{
		vector[i] = arr[i - vectorSize];
	}
}

template <int N, class T>
template <int M>
__device__ __host__
inline Vector<N, T>::Vector(const Vector<M, T>& other)
{
	static_assert(M < N, "Cannot copy large vector into small vector");
	UNROLL_LOOP
	for(int i = 0; i < M; i++)
	{
		vector[i] = other[i];
	}
	UNROLL_LOOP
	for(int i = M; i < N; i++)
	{
		vector[i] = 0.0f;
	}
}

template <int N, class T>
__device__ __host__
inline Vector<N, T>::operator float*()
{
	return vector;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T>::operator const float *() const
{
	return vector;
}

template <int N, class T>
__device__ __host__
inline float& Vector<N, T>::operator[](int i)
{
	return vector[i];
}

template <int N, class T>
__device__ __host__
inline const float& Vector<N, T>::operator[](int i) const
{
	return vector[i];
}

template <int N, class T>
__device__ __host__
inline void Vector<N, T>::operator+=(const Vector& right)
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		vector[i] += right[i];
	}
}

template <int N, class T>
__device__ __host__
inline void Vector<N, T>::operator-=(const Vector& right)
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		vector[i] -= right[i];
	}
}

template <int N, class T>
__device__ __host__
inline void Vector<N, T>::operator*=(const Vector& right)
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		vector[i] *= right[i];
	}
}

template <int N, class T>
__device__ __host__
inline void Vector<N, T>::operator*=(float right)
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		vector[i] *= right;
	}
}

template <int N, class T>
__device__ __host__
inline void Vector<N, T>::operator/=(const Vector& right)
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		vector[i] /= right[i];
	}
}

template <int N, class T>
__device__ __host__
inline void Vector<N, T>::operator/=(float right)
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		vector[i] /= right;
	}
}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::operator+(const Vector& right) const
{
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		v[i] = vector[i] + right[i];
	}
	return v;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::operator-(const Vector& right) const
{
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		v[i] = vector[i] - right[i];
	}
	return v;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::operator-() const
{
	Vector<N, T> v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		v[i] = -vector[i];
	}
	return v;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::operator*(const Vector& right) const
{
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		v[i] = vector[i] * right[i];
	}
	return v;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::operator*(float right) const
{
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		v[i] = vector[i] * right;
	}
	return v;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::operator/(const Vector& right) const
{
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		v[i] = vector[i] / right[i];
	}
	return v;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::operator/(float right) const
{
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		v[i] = vector[i] / right;
	}
	return v;
}

template <int N, class T>
__device__ __host__
inline bool Vector<N, T>::operator==(const Vector& right) const
{
	bool b = true;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		b &= vector[i] == right[i];
	}
	return b;
}

template <int N, class T>
__device__ __host__
inline bool Vector<N, T>::operator!=(const Vector& right) const
{
	return !(*this == right);
}

template <int N, class T>
__device__ __host__
inline float Vector<N, T>::Dot(const Vector& right) const
{
	float data = 0;
	for(int i = 0; i < N; i++)
	{
		data += vector[i] * right[i];
	}
	return data;
}

template <int N, class T>
__device__ __host__
inline float Vector<N, T>::Length() const
{
	return std::sqrt(LengthSqr());
}

template <int N, class T>
__device__ __host__
inline float Vector<N, T>::LengthSqr() const
{
	return Dot(*this);
}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::Normalize() const
{
	float lengthInv = 1.0f / Length();
	
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		v[i] = vector[i] * lengthInv;
	}
	return v;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T>& Vector<N, T>::NormalizeSelf()
{
	float lengthInv = 1.0f / Length();	
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		vector[i] *= lengthInv;
	}
	return *this;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::Clamp(const Vector& min, const Vector& max) const
{
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		#ifdef __CUDA_ARCH__
			v[i] = fminf(fmaxf(min[i], vector[i]), max[i]);
		#else 
			v[i] = std::min(std::max(min[i], vector[i]), max[i]);
		#endif
		
	}
	return v;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::Clamp(float min, float max) const
{
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		#ifdef __CUDA_ARCH__
			v[i] = fminf(fmaxf(min, vector[i]), max);
		#else 
			v[i] = std::min(std::max(min, vector[i]), max);
		#endif
		
	}
	return v;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T>& Vector<N, T>::ClampSelf(const Vector& min, const Vector& max)
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		#ifdef __CUDA_ARCH__
			vector[i] = fminf(fmaxf(min[i], vector[i]), max[i]);
		#else 
			vector[i] = std::min(std::max(min[i], vector[i]), max[i]);
		#endif	
	}
	return *this;
}

template <int N, class T>
__device__ __host__
inline Vector<N, T>& Vector<N, T>::ClampSelf(float min, float max)
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		#ifdef __CUDA_ARCH__
			vector[i] = fminf(fmaxf(min, vector[i]), max);
		#else 
			vector[i] = std::min(std::max(min, vector[i]), max);
		#endif		
	}
	return *this;
}

template <int N, class T>
__device__ __host__ 
Vector<N, T> Vector<N, T>::Abs() const
{
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		#ifdef __CUDA_ARCH__
			v[i] = fabsf(vector[i]);
		#else 
			v[i] = std::abs(vector[i]);
		#endif	
	}
	return v;
}

template <int N, class T>
__device__ __host__ 
Vector<N, T>& Vector<N, T>::AbsSelf()
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		#ifdef __CUDA_ARCH__
			vector[i] = fabsf(vector[i]);
		#else 
			vector[i] = std::abs(vector[i]);
		#endif			
	}
	return *this;
}

template <int N, class T>
__device__ __host__ 
Vector<N, T> Vector<N, T>::Round() const
{

}

template <int N, class T>
__device__ __host__ 
Vector<N, T>& Vector<N, T>::RoundSelf()
{

}

template <int N, class T>
__device__ __host__ 
Vector<N, T> Vector<N, T>::Floor() const
{

}

template <int N, class T>
__device__ __host__ 
Vector<N, T>& Vector<N, T>::FloorSelf()
{

}

template <int N, class T>
__device__ __host__ 
Vector<N, T> Vector<N, T>::Ceil() const
{

}

template <int N, class T>
__device__ __host__ 
Vector<N, T>& Vector<N, T>::CeilSelf()
{

}

template <int N, class T>
__device__ __host__ 
Vector<N, T> Vector<N, T>::Min(const Vector&, const Vector&)
{

}

template <int N, class T>
__device__ __host__ 
Vector<N, T> Vector<N, T>::Min(const Vector&, float)
{

}

template <int N, class T>
__device__ __host__ 
Vector<N, T> Vector<N, T>::Max(const Vector&, const Vector&)
{

}

template <int N, class T>
__device__ __host__ 
Vector<N, T> Vector<N, T>::Max(const Vector&, float)
{

}

template <int N, class T>
__device__ __host__
inline Vector<N, T> Vector<N, T>::Lerp(const Vector& v0, const Vector& v1, float t)
{
	assert(t >= 0.0f && t <= 1.0f);
	Vector v;
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		v[i] = (1.0f - t) * v0[i] + t * v1[i];
	}
	return v;
}

template<int N, class T>
__device__ __host__
inline Vector<N, T> operator*(float left, const Vector<N, T>& vec)
{
	return vec * left;
}

// Cross product (only for 3d vectors)
__device__ __host__
inline Vector3 Cross(const Vector3& v0, const Vector3& v1)
{
	Vector3 result(v0[1] * v1[2] - v0[2] * v1[1],
				   v0[2] * v1[0] - v0[0] * v1[2],
				   v0[0] * v1[1] - v0[1] * v1[0]);
	return result;
}