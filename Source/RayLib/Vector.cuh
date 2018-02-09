//#pragma once
//
//// Constructors & Destructor
//template <int N>
//constexpr __device__ __host__ 
//inline Vector<N>::Vector()
//{
//	#pragma unroll
//	for(int i = 0; < N; i++)
//	{
//		vector[i] = 0.0f;
//	}
//}
//
//template <int N>
//__device__ __host__ 
//inline Vector<N>::Vector(float f)
//{
//	//#pragma unroll
//	for(int i = 0; i < N; i++)
//	{
//		vector[i] = f;
//	}
//}
//
//template <int N>
//__device__ __host__ 
//inline Vector<N>::Vector(const float data[N])
//{
//	#pragma unroll
//	for(int i = 0; i < N; i++)
//	{
//		vector[i] = data[i];
//	}
//}
//
//template <int N>
//template <class... Args>
//__device__ __host__ 
//inline Vector<N>::Vector(const Args... dataList)
//	: vector{static_cast<float>(dataList) ...}
//{
//	static_assert(sizeof...(dataList) == N, "Vector constructor should have exact "
//				  "same count of template count "
//				  "as arguments");
//}
//
//template <int N>
//template <int M>
//__device__ __host__
//inline Vector<N>::Vector(const Vector<M>& other)
//{
//	
//}
//
//template <int N>
//__device__ __host__
//inline Vector<N>::operator float*()
//{
//	return vector;
//}
//
//template <int N>
//
//inline Vector<N>::operator const float *() const
//{
//	return vector;
//}
//
//template <int N>
//__device__ __host__
//inline float& Vector<N>::operator[](int i)
//{
//	return vector[i];
//}
//
//template <int N>
//__device__ __host__
//inline const float& Vector<N>::operator[](int i) const
//{
//	return vector[i];
//}