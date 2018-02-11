#pragma once

/**

Arbitrary sized square matrix. 

N should be 2, 3, 4 at most.

*/

#include <algorithm>
#include <type_traits>
#include "CudaCheck.h"
#include "Vector.h"

template<class T>
using ArithmeticEnable = typename std::enable_if<std::is_arithmetic<T>::value>::type;

template<int N, class T, typename = ArithmeticEnable<T>>
class Matrix;

template<int N, class T>
class Matrix<N, T>
{
	static_assert(N == 2 || N == 3 || N == 4, "Matrix size should be 2x2, 3x3 or 4x4");

	private:
		T									vector[N*N];

	protected:
	public:
		// Constructors & Destructor
		constexpr __device__ __host__		Matrix();
		constexpr __device__ __host__		Matrix(float);
		constexpr __device__ __host__		Matrix(const float* data);
		template <class... Args>
		constexpr __device__ __host__		Matrix(const Args... dataList);
		template <int M>
		__device__ __host__					Matrix(const Matrix<M, T>&);
		~Matrix() = default;
		

		// MVC bug? these trigger std::trivially_copyable static assert
		//									Matrix(const Matrix&) = default;
		//Matrix&							operator=(const Matrix&) = default;

		// Accessors
		__device__ __host__	explicit		operator float*();
		__device__ __host__	explicit		operator const float *() const;
		__device__ __host__ float&			operator[](int);
		__device__ __host__ const float&	operator[](int) const;

		// Modify
		__device__ __host__ void			operator+=(const Matrix&);
		__device__ __host__ void			operator-=(const Matrix&);
		__device__ __host__ void			operator*=(const Matrix&);
		__device__ __host__ void			operator*=(float);
		__device__ __host__ void			operator/=(const Matrix&);
		__device__ __host__ void			operator/=(float);

		__device__ __host__ Matrix			operator+(const Matrix&) const;
		__device__ __host__ Matrix			operator-(const Matrix&) const;
		__device__ __host__ Matrix			operator-() const;
		__device__ __host__ Matrix			operator/(const Matrix&) const;
		__device__ __host__ Matrix			operator/(float) const;

		__device__ __host__ Matrix			operator*(const Matrix&) const;
		__device__ __host__ Vector<N, T>	operator*(const Vector<N, T>&) const;
		__device__ __host__ Matrix			operator*(float) const;

		// Logic
		__device__ __host__ bool			operator==(const Matrix&) const;
		__device__ __host__ bool			operator!=(const Matrix&) const;

		// Utilty	
		__device__ __host__ float			Determinant() const;
		__device__ __host__ Matrix			Inverse() const;
		__device__ __host__ Matrix&			InverseSelf();
		__device__ __host__ Matrix			Transpose() const;
		__device__ __host__ Matrix			TransposeSelf();
		__device__ __host__ Matrix			Clamp(const Matrix&, const Matrix&) const;
		__device__ __host__ Matrix			Clamp(float min, float max) const;
		__device__ __host__ Matrix&			ClampSelf(const Matrix&, const Matrix&);
		__device__ __host__ Matrix&			ClampSelf(float min, float max);
		__device__ __host__ Matrix			Abs() const;
		__device__ __host__ Matrix&			AbsSelf();
		__device__ __host__ Matrix			Round() const;
		__device__ __host__ Matrix&			RoundSelf();
		__device__ __host__ Matrix			Floor() const;
		__device__ __host__ Matrix&			FloorSelf();
		__device__ __host__ Matrix			Ceil() const;
		__device__ __host__ Matrix&			CeilSelf();

		static __device__ __host__ Matrix	Min(const Matrix&, const Matrix&);
		static __device__ __host__ Matrix	Min(const Matrix&, float);
		static __device__ __host__ Matrix	Max(const Matrix&, const Matrix&);
		static __device__ __host__ Matrix	Max(const Matrix&, float);
		static __device__ __host__ Matrix	Lerp(const Matrix&, const Matrix&, float);

		// Transfor Matrix Generation
		static __device__ __host__ Matrix	Translate(const IEVector3&);
		static __device__ __host__ Matrix	Scale(float);
		static __device__ __host__ Matrix	Scale(float x, float y, float z);
		static __device__ __host__ Matrix	Rotate(float angle, const Vector3&);
		static __device__ __host__ Matrix	Rotate(const IEQuaternion&);
		static __device__ __host__ Matrix	Perspective(float fovXRadians, float aspectRatio,
														float nearPlane, float farPlane);
		static __device__ __host__ Matrix	Ortogonal(float left, float right,
													  float top, float bottom,
													  float nearPlane, float farPlane);
		static __device__ __host__ Matrix	Ortogonal(float width, float height,
													  float nearPlane, float farPlane);
		static __device__ __host__ Matrix	LookAt(const Vector3& eyePos,
												   const Vector3& at,
												   const Vector3& up);
};

// Left Scalar operators
template<int N, class T>
static __device__ __host__ Matrix<N, T> operator*(float, const Matrix<N,T>&);

// Typeless vectors are defaulted to float
using Matrix2x2 = Matrix<2, float>;
using Matrix3x3 = Matrix<3, float>;
using Matrix4x4 = Matrix<4, float>;
// Float Type
using Matrix2x2f = Matrix<2, float>;
using Matrix3x3f = Matrix<3, float>;
using Matrix4x4f = Matrix<4, float>;
// Double Type
using Matrix2x2d = Matrix<2, double>;
using Matrix3x3d = Matrix<3, double>;
using Matrix4x4d = Matrix<4, double>;
// Integer Type
using Matrix2x2i = Matrix<2, int>;
using Matrix3x3i = Matrix<3, int>;
using Matrix4x4i = Matrix<4, int>;
// Unsigned Integer Type
using Matrix2x2ui = Matrix<2, unsigned int>;
using Matrix3x3ui = Matrix<3, unsigned int>;
using Matrix4x4ui = Matrix<4, unsigned int>;

// Requirements of Vectors
static_assert(std::is_literal_type<Matrix3x3>::value == true, "Matrices has to be literal types");
static_assert(std::is_trivially_copyable<Matrix3x3>::value == true, "Matrices has to be trivially copyable");
static_assert(std::is_polymorphic<Matrix3x3>::value == false, "Matrices should not be polymorphic");

// Special 4x4 Matrix Operation
static __device__ __host__ Vector3 ExtractScaleInfo(const Matrix4x4&);

#include "Matrix.hpp"	// CPU & GPU