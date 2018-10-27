#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "Constants.h"
#include "Vector.h"

namespace CosineDist
{
	// Hemi
	template <class T, class = FloatEnable<T>>
	__device__ __host__
	inline T HemiPDF(T cosTetha)
	{
		return cos(cosTetha) * static_cast<T>(MathConstants::Pi_d);
	}

	template <class T, class = FloatEnable<T>>
	__device__ __host__
	inline Vector<3, T> HemiICDF(const Vector<2, T>& xi)
	{
		T xi1Coeff = 2 * static_cast<T>(MathConstants::Pi_d) * xi[1];
		Vector<3,T> dir;
		dir[0] = sqrt(xi[0]) * cos(xi1Coeff);
		dir[1] = sqrt(xi[0]) * sin(xi1Coeff);
		dir[2] = sqrt(1.0f - dir[0] * dir[0] - dir[1] * dir[1]);
		return dir;
	}
}