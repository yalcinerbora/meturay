#pragma once
/**

Quaternion, layout is (w, x, y, z)
where; v[0] = w, v[1] = x, v[2] = y, v[3] = z

*/

#include "Vector.h"
#include "Constants.h"

template<class T>
using QuatFloatEnable = typename std::enable_if<std::is_floating_point<T>::value>::type;

template<class T, typename = QuatFloatEnable<T>>
class Quaternion;

template<class T>
class Quaternion<T>
{
    private:
    Vector<4, T>                vec;

    protected:

    public:
    // Constructors & Destructor
    constexpr                   Quaternion() = default;
    constexpr HYBRD_FUNC        Quaternion(T w, T x, T y, T z);
    constexpr HYBRD_FUNC        Quaternion(const T*);
    HYBRD_FUNC                  Quaternion(T angle, const Vector<3, T>& axis);
    HYBRD_FUNC                  Quaternion(const Vector<4, T>& vec);
                                Quaternion(const Quaternion&) = default;
                                ~Quaternion() = default;
    Quaternion&                 operator=(const Quaternion&) = default;

    //
    HYBRD_FUNC explicit         operator Vector<4, T>& ();
    HYBRD_FUNC explicit         operator const Vector<4, T>& () const;
    HYBRD_FUNC explicit         operator T* ();
    HYBRD_FUNC explicit         operator const T* () const;
    HYBRD_FUNC T&               operator[](int);
    HYBRD_FUNC const T&         operator[](int) const;

    // Operators
    HYBRD_FUNC Quaternion       operator*(const Quaternion&) const;
    HYBRD_FUNC Quaternion       operator*(T) const;
    HYBRD_FUNC Quaternion       operator+(const Quaternion&) const;
    HYBRD_FUNC Quaternion       operator-(const Quaternion&) const;
    HYBRD_FUNC Quaternion       operator-() const;
    HYBRD_FUNC Quaternion       operator/(T) const;

    HYBRD_FUNC void             operator*=(const Quaternion&);
    HYBRD_FUNC void             operator*=(T);
    HYBRD_FUNC void             operator+=(const Quaternion&);
    HYBRD_FUNC void             operator-=(const Quaternion&);
    HYBRD_FUNC void             operator/=(T);

    // Logic
    HYBRD_FUNC bool             operator==(const Quaternion&) const;
    HYBRD_FUNC bool             operator!=(const Quaternion&) const;

    // Utility
    HYBRD_FUNC [[nodiscard]]
    Quaternion                  Normalize() const;
    HYBRD_FUNC Quaternion&      NormalizeSelf();
    HYBRD_FUNC T                Length() const;
    HYBRD_FUNC T                LengthSqr() const;
    HYBRD_FUNC [[nodiscard]]
    Quaternion                  Conjugate() const;
    HYBRD_FUNC Quaternion&      ConjugateSelf();
    HYBRD_FUNC T                Dot(const Quaternion&) const;
    HYBRD_FUNC Vector<3, T>     ApplyRotation(const Vector<3, T>&) const;
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
HYBRD_FUNC
Quaternion<T> operator*(T, const Quaternion<T>&);

// Static Utility
namespace Quat
{
    template <class T>
    HYBRD_FUNC Quaternion<T>    NLerp(const Quaternion<T>& start, const Quaternion<T>& end, T t);
    template <class T>
    HYBRD_FUNC Quaternion<T>    SLerp(const Quaternion<T>& start, const Quaternion<T>& end, T t);
    template <class T>
    HYBRD_FUNC Quaternion<T>    BarySLerp(const Quaternion<T>& q0,
                                          const Quaternion<T>& q1,
                                          const Quaternion<T>& q2,
                                          T a, T b);
    template <class T>
    HYBRD_FUNC Quaternion<T>    RotationBetween(const Vector<3, T>& a,
                                                const Vector<3, T>& b);
    template <class T>
    HYBRD_FUNC Quaternion<T>    RotationBetweenZAxis(const Vector<3, T>& b);
}

namespace TransformGen
{
    template <class T>
    HYBRD_FUNC void             Space(Quaternion<T>&,
                                      const Vector<3, T>& x,
                                      const Vector<3, T>& y,
                                      const Vector<3, T>& z);
    template <class T>
    HYBRD_FUNC void             InvSpace(Quaternion<T>&,
                                         const Vector<3, T>& x,
                                         const Vector<3, T>& y,
                                         const Vector<3, T>& z);
}

// Implementation
#include "Quaternion.hpp"

// Constants
static constexpr QuatF IdentityQuatF = QuatF(1.0f, 0.0f, 0.0f, 0.0f);
static constexpr QuatD IdentityQuatD = QuatD(1.0, 0.0, 0.0, 0.0);

// Quaternion Traits
template<class T>
struct IsQuatType
{
    static constexpr bool value =
        std::is_same<T, QuatF>::value ||
        std::is_same<T, QuatD>::value;
};

// // Quaternion Extern
// extern template class Quaternion<float>;
// extern template class Quaternion<double>;