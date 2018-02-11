#pragma once

/**

Arbitrary sized vector. Vector is column vector (N x 1 matrix)
which means that it can only be multipled with matrices from right.

N should be 2, 3, 4 at most.

*/

#include <algorithm>
#include <type_traits>
#include "CudaCheck.h"

//template <int N, class T, typename Enable = void>
//class Vector
//{};

template<class T>
using ArithmeticEnable = typename std::enable_if<std::is_arithmetic<T>::value>::type;

template<int N, class T, typename = ArithmeticEnable<T>>
class Vector;

template<int N, class T>
class Vector<N, T>
{
	static_assert(N == 2 || N == 3 || N == 4, "Vector size should be 2, 3 or 4");

	private:
		T									vector[N];

	protected:
	public:
		// Constructors & Destructor
		constexpr __device__ __host__		Vector();
		constexpr __device__ __host__		Vector(float);
		constexpr __device__ __host__		Vector(const float* data);
		template <class... Args>
		constexpr __device__ __host__		Vector(const Args... dataList);
		template <int M>
		__device__ __host__					Vector(const Vector<M, T>&);
											~Vector() = default;

		// MVC bug? these trigger std::trivially_copyable static assert
										Vector(const Vector&) = default;
		Vector&							operator=(const Vector&) = default;

		// Accessors
		__device__ __host__	explicit		operator float*();
		__device__ __host__	explicit		operator const float *() const;
		__device__ __host__ float&			operator[](int);
		__device__ __host__ const float&	operator[](int) const;

		// Modify
		__device__ __host__ void			operator+=(const Vector&);
		__device__ __host__ void			operator-=(const Vector&);
		__device__ __host__ void			operator*=(const Vector&);
		__device__ __host__ void			operator*=(float);
		__device__ __host__ void			operator/=(const Vector&);
		__device__ __host__ void			operator/=(float);

		__device__ __host__ Vector			operator+(const Vector&) const;
		__device__ __host__ Vector			operator-(const Vector&) const;
		__device__ __host__ Vector			operator-() const;
		__device__ __host__ Vector			operator*(const Vector&) const;
		__device__ __host__ Vector			operator*(float) const;
		__device__ __host__ Vector			operator/(const Vector&) const;
		__device__ __host__ Vector			operator/(float) const;

		// Logic
		__device__ __host__ bool			operator==(const Vector&) const;
		__device__ __host__ bool			operator!=(const Vector&) const;

		// Utilty
		__device__ __host__ float			Dot(const Vector&) const;
		__device__ __host__ float			Length() const;
		__device__ __host__ float			LengthSqr() const;
		__device__ __host__ Vector			Normalize() const;
		__device__ __host__ Vector&			NormalizeSelf();
		__device__ __host__ Vector			Clamp(const Vector&, const Vector&) const;
		__device__ __host__ Vector			Clamp(float min, float max) const;
		__device__ __host__ Vector&			ClampSelf(const Vector&, const Vector&);
		__device__ __host__ Vector&			ClampSelf(float min, float max);
		__device__ __host__ Vector			Abs() const;
		__device__ __host__ Vector&			AbsSelf();
		__device__ __host__ Vector			Round() const;
		__device__ __host__ Vector&			RoundSelf();
		__device__ __host__ Vector			Floor() const;
		__device__ __host__ Vector&			FloorSelf();
		__device__ __host__ Vector			Ceil() const;
		__device__ __host__ Vector&			CeilSelf();

		static __device__ __host__ Vector	Min(const Vector&, const Vector&);
		static __device__ __host__ Vector	Min(const Vector&, float);
		static __device__ __host__ Vector	Max(const Vector&, const Vector&);
		static __device__ __host__ Vector	Max(const Vector&, float);		
		static __device__ __host__ Vector	Lerp(const Vector&, const Vector&, float);
};

// Left scalars
template<int N, class T>
static __device__ __host__ Vector<N,T> operator*(float, const Vector<N, T>&);

// Typeless vectors are defaulted to float
using Vector2 = Vector<2, float>;
using Vector3 = Vector<3, float>;
using Vector4 = Vector<4, float>;
// Float Type
using Vector2f = Vector<2, float>;
using Vector3f = Vector<3, float>;
using Vector4f = Vector<4, float>;
// Double Type
using Vector2d= Vector<2, double>;
using Vector3d= Vector<3, double>;
using Vector4d= Vector<4, double>;
// Integer Type
using Vector2i = Vector<2, int>;
using Vector3i = Vector<3, int>;
using Vector4i = Vector<4, int>;
// Unsigned Integer Type
using Vector2ui = Vector<2, unsigned int>;
using Vector3ui = Vector<3, unsigned int>;
using Vector4ui = Vector<4, unsigned int>;

// Requirements of Vectors
static_assert(std::is_literal_type<Vector3>::value == true, "Vectors has to be literal types");
static_assert(std::is_trivially_copyable<Vector3>::value == true, "Vectors has to be trivially copyable");
static_assert(std::is_polymorphic<Vector3>::value == false, "Vectors should not be polymorphic");

// Cross product (only for 3d vectors)
static __device__ __host__ Vector3 Cross(const Vector3&, const Vector3&);

// Implementation
#include "Vector.hpp"	// CPU & GPU

// Basic Constants
static constexpr Vector3 XAxis = Vector3(1.0f, 0.0f, 0.0f);
static constexpr Vector3 YAxis = Vector3(0.0f, 1.0f, 0.0f);
static constexpr Vector3 ZAxis = Vector3(0.0f, 0.0f, 1.0f);

static constexpr Vector2 Zero2 = Vector2(0.0f, 0.0f);
static constexpr Vector3 Zero3 = Vector3(0.0f, 0.0f, 0.0f);
static constexpr Vector4 Zero4 = Vector4(0.0f, 0.0f, 0.0f, 0.0f);
