#pragma once

/**

Arbitrary sized vector. Vector is column vector
which means that it can only be multipled with matrices from right.

S should be 2, 3, 4 at most.

*/

#include <algorithm>
#include "CudaCheck.h"

template<int N>
class Vector
{
	static_assert(N == 2 || N == 3 || N == 4, "Vector size should be 2, 3 or 4");

	private:
		float							vector[N];

	protected:
	public:
		// Constructors & Destructor
		constexpr __device__ __host__	Vector();
		__device__ __host__				Vector(float);
		__device__ __host__				Vector(const float data[N]);
		template <class... Args>
		__device__ __host__				Vector(const Args... dataList);
		template <int M>
		__device__ __host__				Vector(const Vector<M>&);
		__device__ __host__				~Vector() = default;

		// MVC bug? these trigger std::trivially_copyable static assert
		//__device__ __host__					Vector(const Vector<N>&) = default;
		//__device__ __host__ Vector<S>&		operator=(const Vector<N>) = default;

		// Accessors
		__device__ __host__	explicit	operator float*();
		__device__ __host__	explicit	operator const float *() const;
		float&							operator[](int);
		const float&					operator[](int) const;

		// Modify
		__device__ __host__ void		operator+=(const Vector&);
		__device__ __host__ void		operator-=(const Vector&);
		__device__ __host__ void		operator*=(const Vector&);
		__device__ __host__ void		operator*=(float);
		__device__ __host__ void		operator/=(const Vector&);
		__device__ __host__ void		operator/=(float);

		__device__ __host__ Vector		operator+(const Vector&) const;
		__device__ __host__ Vector		operator-(const Vector&) const;
		__device__ __host__ Vector		operator-() const;
		__device__ __host__ Vector		operator*(const Vector&) const;
		__device__ __host__ Vector		operator*(float) const;
		__device__ __host__ Vector		operator/(const Vector&) const;
		__device__ __host__ Vector		operator/(float) const;

		// Logic
		__device__ __host__ bool		operator==(const Vector&) const;
		__device__ __host__ bool		operator!=(const Vector&) const;

		// Utilty
		__device__ __host__ float		DotProduct(const Vector&) const;
		__device__ __host__ float		Length() const;
		__device__ __host__ float		LengthSqr() const;
		__device__ __host__ Vector		Normalize() const;
		__device__ __host__ Vector&		NormalizeSelf();
		__device__ __host__ Vector		Clamp(const Vector&, const Vector&) const;
		__device__ __host__ Vector		Clamp(float min, float max) const;
		__device__ __host__ Vector&		ClampSelf(const Vector&, const Vector&);
		__device__ __host__ Vector&		ClampSelf(float min, float max);
};

// Left scalars
template<class Vector>
static Vector operator*(float, const Vector&);

// Convenience
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;
using Vector4 = Vector<4>;

// Requirements of Vectors
static_assert(std::is_literal_type<Vector3>::value == true, "Vectors has to be literal types");
static_assert(std::is_trivially_copyable<Vector3>::value == true, "Vectors has to be trivially copyable");
static_assert(std::is_polymorphic<Vector3>::value == false, "Vectors should not be polymorphic");

// Cross product (only for 3d vectors)
static __device__ __host__ Vector3 CrossProduct(Vector3&);

// Implementations
#include "Vector.hpp"	// CPU
#include "Vector.cuh"	// GPU (CUDA)

// Basic Constants
//static const Vector3 XAxis = Vector3({ 1.0f, 0.0f, 0.0f });
