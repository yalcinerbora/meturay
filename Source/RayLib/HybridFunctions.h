#pragma once

#include "CudaCheck.h"

namespace HybridFuncs
{
	template <class T>
	__device__ __host__ void Swap(T&, T&);
}
 
template <class T>
__device__ __host__
void HybridFuncs::Swap(T& t0, T& t1)
{
	T temp = t0;
	t0 = t1;
	t1 = temp;
}