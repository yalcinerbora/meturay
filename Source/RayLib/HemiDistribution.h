#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "Constants.h"
#include "Vector.h"

namespace HemiDistribution
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
    inline Vector<3, T> HemiCosineCDF(const Vector<2, T>& xi)
    {
        T xi1Coeff = 2 * static_cast<T>(MathConstants::Pi_d) * xi[1];
        Vector<3,T> dir;
        dir[0] = sqrt(xi[0]) * cos(xi1Coeff);
        dir[1] = sqrt(xi[0]) * sin(xi1Coeff);
        dir[2] = sqrt(1 - dir[0] * dir[0] - dir[1] * dir[1]);
        return dir;
    }

    template <class T, class = FloatEnable<T>>
    __device__ __host__
    inline Vector<3, T> HemiUniformCDF(const Vector<2, T>& xi)
    {
        T xi0Coeff = 1 - xi[0] * xi[0];
        T xi1Coeff = 2 * static_cast<T>(MathConstants::Pi_d) * xi[1];
        Vector<3,T> dir;
        dir[0] = sqrt(xi0Coeff) * cos(xi1Coeff);
        dir[1] = sqrt(xi0Coeff) * sin(xi1Coeff);
        dir[2] = xi[0];
        return dir;
    }
}