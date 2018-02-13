#pragma once

template <int N, class T>
__device__ __host__
inline Matrix<N, T>::Matrix()
{

}

template <int N, class T>
__device__ __host__	
inline Matrix<N, T>::Matrix(float)
{

}

template <int N, class T>
__device__ __host__
inline Matrix<N, T>::Matrix(const float* data)
{

}

template <int N, class T>
template <class... Args, typename = AllArithmeticEnable<Args...>>
__device__ __host__
inline constexpr Matrix<N,T>::Matrix(const Args... dataList)
{

}

template <int N, class T>
__device__ __host__
inline Matrix<N, T>::Matrix(const Vector<N, T> columns[N])
{

}

template <int N, class T>
template <int M>
__device__ __host__
inline Matrix<N, T>::Matrix(const Matrix<M, T>&)
{

}

template <int N, class T>
__device__ __host__	
inline explicit Matrix<N, T>::operator float*()
{

}

template <int N, class T>
__device__ __host__	
inline explicit Matrix<N, T>::operator const float *() const
{

}

template <int N, class T>
__device__ __host__ 
inline T& Matrix<N, T>::operator[](int)
{

}

template <int N, class T>
__device__ __host__ 
inline const T& Matrix<N, T>::operator[](int) const
{

}

template <int N, class T>
__device__ __host__ 
inline T& Matrix<N, T>::operator()(int, int)
{

}

template <int N, class T>
__device__ __host__ 
inline const T& Matrix<N, T>::operator()(int, int) const
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator+=(const Matrix&)
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator-=(const Matrix&)
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator*=(const Matrix&)
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator*=(float)
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator/=(const Matrix&)
{

}

template <int N, class T>
__device__ __host__ 
inline void Matrix<N, T>::operator/=(float)
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator+(const Matrix&) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator-(const Matrix&) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator-() const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator/(const Matrix&) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator/(float) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator*(const Matrix&) const
{

}

template <int N, class T>
__device__ __host__ 
inline Vector<N, T>	Matrix<N, T>::operator*(const Vector<N, T>&) const
{

}

template <int N, class T>
__device__ __host__ 
inline Matrix Matrix<N, T>::operator*(float) const
{

}

template <int N, class T>
__device__ __host__ 
inline bool Matrix<N, T>::operator==(const Matrix&) const
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