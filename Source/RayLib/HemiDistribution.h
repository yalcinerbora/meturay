#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "Constants.h"
#include "Vector.h"

namespace HemiDistribution
{
    template <class T, class = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Vector<3, T> HemiCosineCDF(const Vector<2, T>& xi, float& pdf)
    {
        T xi1Coeff = 2 * static_cast<T>(MathConstants::Pi_d) * xi[1];
        Vector<3, T> dir;
        dir[0] = sqrt(xi[0]) * cos(xi1Coeff);
        dir[1] = sqrt(xi[0]) * sin(xi1Coeff);
        dir[2] = sqrt(max((T)0, (1 - dir[0] * dir[0]
                                 - dir[1] * dir[1])));

      // Sampling from unit hemisphere
      // Normal is (0,0,1) pdf is NdotD / Pi
      // Thus return D.z / Pi
        pdf = dir[2] * static_cast<T>(MathConstants::InvPi_d);
        return dir;
    }

    template <class T, class = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Vector<3, T> HemiUniformCDF(const Vector<2, T>& xi, float& pdf)
    {
        T xi0Coeff = sqrt(max((T)0, 1 - xi[0] * xi[0]));
        T xi1Coeff = 2 * static_cast<T>(MathConstants::Pi_d) * xi[1];
        Vector<3, T> dir;
        dir[0] = xi0Coeff * cos(xi1Coeff);
        dir[1] = xi0Coeff * sin(xi1Coeff);
        dir[2] = xi[0];
        // PDF is static 1 / (2 * Pi)
        pdf = static_cast<T>(MathConstants::InvPi_d) * 0.5f;
        return dir;
    }
}