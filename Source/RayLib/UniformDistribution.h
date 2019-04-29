#pragma once

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "Constants.h"
#include "Vector.h"

namespace UniformDist
{
    // 1D Classic
    template <class T, class = FloatEnable<T>>
    __device__ __host__
    inline T PDF(/*T xi,*/ const T start, const T end)
    {
        assert(end > start);
        if(xi < start || xi > end) return 0;
        return 1 / (end - start);
    }

    template <class T, class = FloatEnable<T>>
    __device__ __host__
    inline T CDF(T xi, const T start, const T end)
    {
        assert(end > start);
        if(xi < start) return 0;
        else if(xi >= end) return 1;
        return 1 / (end - start);
    }

    template <class T, class = FloatEnable<T>>
    __device__ __host__
    inline T ICDF(T xi, const T start, const T end)
    {
        assert(end > start);
        return xi * (end - start) + start;
    }

    // Hemispherical
    template <class T, class = FloatEnable<T>>
    __device__ __host__
    inline T HemiPDF(/*const Vector<2,T>& xi*/)
    {
        return static_cast<T>(0.5) * static_cast<T>(MathConstants::InvPi_d);
    }

    template <class T, class = FloatEnable<T>>
    __device__ __host__
    inline Vector<3, T> HemiICDF(const Vector<2, T>& xi)
    {
        // Disk distribution with uniform Z 
        T xi0Coeff = sqrt(1 - xi[0] * xi[0]);
        T xi1Coeff = 2 * static_cast<T>(MathConstants::Pi_d) * xi[1];

        Vector<3, T> dir;
        dir[0] = xi0Coeff * cos(xi1Coeff);
        dir[1] = xi0Coeff * sin(xi1Coeff);
        dir[2] = xi[0];
        return dir;
    }

    // Spherical
    template <class T, class = FloatEnable<T>>
    __device__ __host__
    inline T SphrPDF(/*const Vector<2,T>& xi*/)
    {
        return static_cast<T>(0.25) * static_cast<T>(MathConstants::InvPi_d);
    }

    template <class T, class = FloatEnable<T>>
    __device__ __host__
    inline Vector<3, T> SphrICDF(const Vector<2, T>& xi)
    {
        // Disk distribution with uniform Z 
        T z = 1 - 2 * xi[0];
        T xi1Coeff = 2 * static_cast<T>(MathConstants::Pi_d) * xi[1];
        T radius = sqrt(1 - z * z);

        Vector<3, T> dir;
        dir[0] = xi0Coeff * cos(xi1Coeff);
        dir[1] = xi0Coeff * sin(xi1Coeff);
        dir[2] = z;
        return dir;
    }

    // Cone
    template <class T, class = FloatEnable<T>>
    __device__ __host__
        inline T ConePDF(T aperture)
    {
        return 1 / (2 * static_cast<T>(MathConstants::Pi_d) * (1 - cos(aperture)));
    }

    template <class T, class = FloatEnable<T>>
    __device__ __host__
        inline Vector<3, T> ConeICDF(const Vector<2, T>& xi, T aperture)
    {
        T cosTheta = (1 - xi[0]) + xi[0] * cos(aperture);
        T sinTheta = sqrt(1 - cosTheta * cosTheta);
        T phi = xi[1] * 2 * Pi;

        Vector<3, T> dir;
        dir[0] = cos(phi) * sinTheta;
        dir[1] = sin(phi) * sinTheta;
        dir[2] = cosTheta;
        return dir;
    }
}