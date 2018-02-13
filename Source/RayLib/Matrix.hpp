#pragma once

template <int N, class T>
__device__ __host__
inline Matrix<N, T>::Matrix()
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		UNROLL_LOOP
		for(int j = 0; j < N; j++)
		{
			if(i == j)
				matrix[i * N + j] = 1;
			else
				matrix[i * N + j] = 0;
		}
	}
}

template <int N, class T>
__device__ __host__	
inline Matrix<N, T>::Matrix(float t)
{
	UNROLL_LOOP
	for(int i = 0; i < N*N; i++)
	{
		matrix[i] = t;
	}
}

template <int N, class T>
__device__ __host__
inline Matrix<N, T>::Matrix(const float* data)
{
	UNROLL_LOOP
	for(int i = 0; i < N*N; i++)
	{
		matrix[i] = data[i];
	}
}

template <int N, class T>
template <class... Args, typename = AllArithmeticEnable<Args...>>
__device__ __host__
inline constexpr Matrix<N,T>::Matrix(const Args... dataList)
	: matrix{static_cast<T>(dataList) ...}
{
	static_assert(sizeof...(dataList) == N, "Matrix constructor should have exact "
											"same count of template count "
											"as arguments");
}

template <int N, class T>
__device__ __host__
inline Matrix<N, T>::Matrix(const Vector<N, T> columns[N])
{
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		const Vector<N, T>& vec = columns[i];
		UNROLL_LOOP
		for(int j = 0; j < N; j++)
		{			
			matrix[i * N + j] = vec[j];
		}
	}
}

template <int N, class T>
template <int M>
__device__ __host__
inline Matrix<N, T>::Matrix(const Matrix<M, T>& other)
{
	static_assert(M < N, "Cannot copy large matrix into small matrix");	
	UNROLL_LOOP
	for(int i = 0; i < N; i++)
	{
		UNROLL_LOOP
		for(int j = 0; j < N; j++)
		{			
			if(i < M && j < M)
				matrix[i * N + j] = other[i * M + j];
			else if(i == N && j == N)
				matrix[i * M + j] = 1;
			else 
				matrix[i * M + j] = 0;
		}
	}
}

template <int N, class T>
__device__ __host__	
inline explicit Matrix<N, T>::operator float*()
{
	return matrix;
}

template <int N, class T>
__device__ __host__	
inline explicit Matrix<N, T>::operator const float *() const
{
	return matrix;
}

template <int N, class T>
__device__ __host__ 
inline T& Matrix<N, T>::operator[](int i)
{
	return matrix[i];
}

template <int N, class T>
__device__ __host__ 
inline const T& Matrix<N, T>::operator[](int) const
{
	return matrix[i];
}

template <int N, class T>
__device__ __host__ 
inline T& Matrix<N, T>::operator()(int row, int column)
{
	return matrix[row * N + column];
}

template <int N, class T>
__device__ __host__ 
inline const T& Matrix<N, T>::operator()(int row, int column) const
{
	return matrix[row * N + column];
}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator+=(const Matrix& right)
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator-=(const Matrix& right)
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator*=(const Matrix& right)
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator*=(float right)
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator/=(const Matrix& right)
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator/=(float)
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator+(const Matrix& right) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator-(const Matrix& right) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator-() const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator/(const Matrix& right) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator/(float right) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator*(const Matrix& right) const
{

}

template <int N, class T>
__device__ __host__ 
inline Vector<N, T>	Matrix<N, T>::operator*(const Vector<N, T>& right) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator*(float right) const
{

}

template <int N, class T>
__device__ __host__ 
inline bool Matrix<N, T>::operator==(const Matrix& right) const
{

}

template <int N, class T>
__device__ __host__ 
inline bool Matrix<N, T>::operator!=(const Matrix&) const
{

}

template <int N, class T>
__device__ __host__ 
inline T Matrix<N, T>::Determinant() const
{

}

template <int N, class T>
template<typename>
__device__ __host__ 
inline Matrix Matrix<N, T>::Inverse() const
{

}

template <int N, class T>
template<typename>
__device__ __host__ 
inline Matrix& Matrix<N, T>::InverseSelf()
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::Transpose() const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::TransposeSelf()
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::Clamp(const Matrix&, const Matrix&) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::Clamp(T min, T max) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix& Matrix<N, T>::ClampSelf(const Matrix&, const Matrix&)
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix& Matrix<N, T>::ClampSelf(T min, T max)
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::Abs() const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix& Matrix<N, T>::AbsSelf()
{

}

template <int N, class T>
template<typename>
__device__ __host__ 
inline Matrix Matrix<N, T>::Round() const
{

}

template <int N, class T>
template<typename>
__device__ __host__ 
inline Matrix& Matrix<N, T>::RoundSelf()
{

}

template <int N, class T>
template<typename>
__device__ __host__ 
inline Matrix Matrix<N, T>::Floor() const
{

}

template <int N, class T>
template<typename>
__device__ __host__ 
inline Matrix& Matrix<N, T>::FloorSelf()
{

}

template <int N, class T>
template<typename>
__device__ __host__ 
inline Matrix Matrix<N, T>::Ceil() const
{

}

template <int N, class T>
template<typename>
__device__ __host__ 
inline Matrix& Matrix<N, T>::CeilSelf()
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::Min(const Matrix&, const Matrix&)
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::Min(const Matrix&, T)
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::Max(const Matrix&, const Matrix&)
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::Max(const Matrix&, T)
{

}

template <int N, class T>
template<typename>
__device__ __host__ 
inline Matrix Matrix<N, T>::Lerp(const Matrix&, const Matrix&, T)
{

}

template<int N, class T>
__device__ __host__ 
Matrix<N, T> operator*(float t, const Matrix<N, T>& mat)
{
	return mat * t;
}

template<class T, typename>
__device__ __host__ 
inline Matrix<4, T> TransformGen::Translate(const Vector<3, T>&)
{

}

template<class T, typename>
__device__ __host__ 
inline Matrix<4, T> TransformGen::Scale(T)
{

}

template<class T, typename>
__device__ __host__ 
inline Matrix<4, T> TransformGen::Scale(T x, T y, T z)
{

}

template<class T, typename>
__device__ __host__ 
inline Matrix<4, T> TransformGen::Rotate(T angle, const Vector<3, T>&)
{

}

template<class T, typename>
__device__ __host__ 
inline Matrix<4, T> TransformGen::Rotate(const Quaternion&)
{

}

template<class T, typename>
__device__ __host__
inline Matrix<4, T> TransformGen::Perspective(T fovXRadians, T aspectRatio,
											  T nearPlane, T farPlane)
{

}

template<class T, typename>
__device__ __host__
inline  Matrix<4, T> TransformGen::Ortogonal(T left, T right,
											 T top, T bottom,
											 T nearPlane, T farPlane)
{

}

template<class T, typename>
__device__ __host__
inline Matrix<4, T> TransformGen::Ortogonal(T width, T height,
											T nearPlane, T farPlane)
{

}

template<class T, typename>
__device__ __host__
inline Matrix<4, T> TransformGen::LookAt(const Vector<3, T>& eyePos,
										 const Vector<3, T>& at,
										 const Vector<3, T>& up)
{

}