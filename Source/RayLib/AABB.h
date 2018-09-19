#pragma once

/**

Arbitrary sized axis aligned bounding box.

N should be 2, 3 or 4 at most.

These are convenience register classes for GPU.

*/

#include "Vector.h"

template<int N, class T, typename = ArithmeticEnable<T>>
class AABB;

template<int N, class T>
class alignas(ChooseVectorAlignment(N * sizeof(T))) AABB<N, T>
{
	private:
		Vector<N, T> min;
		Vector<N, T> max;

		// Constructors & Destructor
		constexpr									AABB() = default;
		__device__ __host__							AABB(const Vector<N, T>& min,
														 const Vector<N, T>& min);
		__device__ __host__							AABB(const T* data);
		template <class... Args, typename =			AllArithmeticEnable<Args...>>
		constexpr __device__ __host__				AABB(const Args... dataList);			
													~AABB() = default;

		// Accessors
		__device__ __host__ const Vector<N, T>&		Min() const;
		__device__ __host__ const Vector<N, T>&		Max() const;
		__device__ __host__ Vector<N, T>			Min();
		__device__ __host__ Vector<N, T>			Max();

		// Mutators
		__device__ __host__ void					SetMin(const Vector<N, T>&);
		__device__ __host__ void					SetMax(const Vector<N, T>&);

		// Functionality
		__device__ __host__ Vector<N, T>			Centroid() const;
};

// Typeless aabbs are defaulted to float
using AABB2 = AABB<2, float>;
using AABB3 = AABB<3, float>;
using AABB4 = AABB<4, float>;
// Float Type
using AABB2f = AABB<2, float>;
using AABB3f = AABB<3, float>;
using AABB4f = AABB<4, float>;
// Double Type
using AABB2d = AABB<2, double>;
using AABB3d = AABB<3, double>;
using AABB4d = AABB<4, double>;

// Requirements of Vectors
static_assert(std::is_literal_type<AABB3>::value == true, "AABBs has to be literal types");
static_assert(std::is_trivially_copyable<AABB3>::value == true, "AABBs has to be trivially copyable");
static_assert(std::is_polymorphic<AABB3>::value == false, "AABBs should not be polymorphic");

// Implementation
#include "AABB.hpp"	// CPU & GPU

// AABB Etern
extern template class AABB<2, float>;
extern template class AABB<3, float>;
extern template class AABB<4, float>;

extern template class AABB<2, double>;
extern template class AABB<3, double>;
extern template class AABB<4, double>;
