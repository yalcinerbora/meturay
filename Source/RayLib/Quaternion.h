#pragma once
/**

Quaternion, layout is (w, x, y, z)
where; v[0] = w, v[1] = x, v[2] = y, v[3] = z

*/

#include "Vector.h"

template<class T>
using QuatFloatEnable = typename std::enable_if<std::is_floating_point<T>::value>::type;

template<class T, typename = QuatFloatEnable<T>>
class Quaternion;

template<class T>
class Quaternion<T>
{
    private:
            Vector<4, T>                        vec;

    protected:

    public:
        // Constructors & Destructor
        constexpr                               Quaternion() = default;
        constexpr __device__ __host__           Quaternion(T w, T x, T y, T z);
        constexpr __device__ __host__           Quaternion(const T*);
        __device__ __host__                     Quaternion(T angle, const Vector<3, T>& axis);
        __device__ __host__                     Quaternion(const Vector<4, T>& vec);
                                                Quaternion(const Quaternion&) = default;
                                                ~Quaternion() = default;
        Quaternion&                             operator=(const Quaternion&) = default;

        //
        __device__ __host__ explicit            operator Vector<4, T>&();
        __device__ __host__ explicit            operator const Vector<4, T>&() const;
        __device__ __host__ explicit            operator T*();
        __device__ __host__ explicit            operator const T*() const;
        __device__ __host__ T&                  operator[](int);
        __device__ __host__ const T&            operator[](int) const;

        // Operators
        __device__ __host__ Quaternion          operator*(const Quaternion&) const;
        __device__ __host__ Quaternion          operator*(T) const;
        __device__ __host__ Quaternion          operator+(const Quaternion&) const;
        __device__ __host__ Quaternion          operator-(const Quaternion&) const;
        __device__ __host__ Quaternion          operator-() const;
        __device__ __host__ Quaternion          operator/(T) const;

        __device__ __host__ void                operator*=(const Quaternion&);
        __device__ __host__ void                operator*=(T);
        __device__ __host__ void                operator+=(const Quaternion&);
        __device__ __host__ void                operator-=(const Quaternion&);
        __device__ __host__ void                operator/=(T);

        // Logic
        __device__ __host__ bool                operator==(const Quaternion&) const;
        __device__ __host__ bool                operator!=(const Quaternion&) const;

        // Utility
        __device__ __host__ Quaternion          Normalize() const;
        __device__ __host__ Quaternion&         NormalizeSelf();
        __device__ __host__ T                   Length() const;
        __device__ __host__ T                   LengthSqr() const;
        __device__ __host__ Quaternion          Conjugate() const;
        __device__ __host__ Quaternion&         ConjugateSelf();
        __device__ __host__ T                   Dot(const Quaternion&) const;
        __device__ __host__ Vector<3,T>         ApplyRotation(const Vector<3, T>&) const;

        // Static Utility
        static __device__ __host__ Quaternion   NLerp(const Quaternion& start, const Quaternion& end, T t);
        static __device__ __host__ Quaternion   SLerp(const Quaternion& start, const Quaternion& end, T t);

        static __device__ __host__ Quaternion   RotationBetween(const Vector<3,T>& a, const Vector<3,T>& b);
        static __device__ __host__ Quaternion   RotationBetweenZAxis(const Vector<3,T>& b);
};

// Quaternion Alias
using QuatF = Quaternion<float>;
using QuatD = Quaternion<double>;

// Requirements of IEQuaternion
static_assert(std::is_trivially_copyable<QuatF>::value == true, "IEQuaternion has to be trivially copyable");
static_assert(std::is_polymorphic<QuatF>::value == false, "IEQuaternion should not be polymorphic");
static_assert(sizeof(QuatF) == sizeof(float) * 4, "IEQuaternion size is not 16 bytes");

// Left Scalar operators
template<class T>
static __device__ __host__ Quaternion<T> operator*(T, const Quaternion<T>&);

// Implementation
#include "Quaternion.hpp"

// Constants
static constexpr QuatF  IdentityQuatF = QuatF(1.0f, 0.0f, 0.0f, 0.0f);
static constexpr QuatD  IdentityQuatD = QuatD(1.0, 0.0, 0.0, 0.0);

// Quaternion Traits
template<class T>
struct IsQuatType
{
    static constexpr bool value =
        std::is_same<T, QuatF>::value ||
        std::is_same<T, QuatD>::value;
};

// Quaternion Extern
extern template class Quaternion<float>;
extern template class Quaternion<double>;