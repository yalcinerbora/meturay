#pragma once

constexpr									AABB() = default;
__device__ __host__							AABB(const Vector<N, T>& min,
												 const Vector<N, T>& max);
__device__ __host__							AABB(const T* data);
template <class... Args, typename = AllArithmeticEnable<Args...>>
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