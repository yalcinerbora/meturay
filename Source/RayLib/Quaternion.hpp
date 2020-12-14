//template<class T>
//__device__ __host__
//inline constexpr Quaternion<T>::Quaternion()
//  : vec(1, 0, 0, 0)
//{}

template<class T>
__device__ __host__
inline constexpr Quaternion<T>::Quaternion(T w, T x, T y, T z)
    : vec(w, x, y, z)
{}

template<class T>
__device__ __host__
inline constexpr Quaternion<T>::Quaternion(const T* v)
    : vec(v)
{}

template<class T>
__device__ __host__
inline Quaternion<T>::Quaternion(T angle, const Vector<3, T>& axis)
{
    angle *= 0.5;
    T sinAngle = sin(angle);

    vec[1] = axis[0] * sinAngle;
    vec[2] = axis[1] * sinAngle;
    vec[3] = axis[2] * sinAngle;
    vec[0] = cos(angle);
}

template<class T>
inline __device__ __host__ Quaternion<T>::Quaternion(const Vector<4, T>& vec)
    : vec(vec)
{}

template<class T>
__device__ __host__
inline Quaternion<T>::operator Vector<4, T>&()
{
    return vec;
}

template<class T>
__device__ __host__
inline Quaternion<T>::operator const Vector<4, T>&() const
{
    return vec;
}

template<class T>
__device__ __host__
inline Quaternion<T>::operator T*()
{
    return static_cast<T*>(vec);
}

template<class T>
__device__ __host__
inline Quaternion<T>::operator const T*() const
{
    return static_cast<const T*>(vec);
}

template<class T>
__device__ __host__
inline T& Quaternion<T>::operator[](int i)
{
    return vec[i];
}

template<class T>
__device__ __host__
inline const T& Quaternion<T>::operator[](int i) const
{
    return vec[i];
}

template<class T>
__device__ __host__
inline Quaternion<T> Quaternion<T>::operator*(const Quaternion& right) const
{
    //return Quaternion(vec[0] * right[0] - vec[1] * right[1] - vec[2] * right[2] - vec[3] * right[3],        // W
    //                  vec[0] * right[1] + vec[1] * right[0] + vec[2] * right[3] - vec[3] * right[2],        // X
    //                  vec[0] * right[2] + vec[2] * right[0] + vec[3] * right[1] - vec[1] * right[3],        // Y
    //                  vec[0] * right[3] + vec[3] * right[0] + vec[1] * right[2] - vec[2] * right[1]);       // Z

    return Quaternion(vec[0] * right[0] - vec[1] * right[1] - vec[2] * right[2] - vec[3] * right[3],    // W
                      vec[0] * right[1] + vec[1] * right[0] + vec[2] * right[3] - vec[3] * right[2],    // X
                      vec[0] * right[2] - vec[1] * right[3] + vec[2] * right[0] + vec[3] * right[1],    // Y
                      vec[0] * right[3] + vec[1] * right[2] - vec[2] * right[1] + vec[3] * right[0]);   // Z
}

template<class T>
__device__ __host__
inline Quaternion<T> Quaternion<T>::operator*(T right) const
{
    return Quaternion<T>(vec * right);
}

template<class T>
__device__ __host__
inline Quaternion<T> Quaternion<T>::operator+(const Quaternion& right) const
{
    return Quaternion(vec + right.vec);
}

template<class T>
__device__ __host__
inline Quaternion<T> Quaternion<T>::operator-(const Quaternion& right) const
{
    return Quaternion(vec - right.vec);
}

template<class T>
__device__ __host__
inline Quaternion<T> Quaternion<T>::operator-() const
{
    return Quaternion(-vec);
}

template<class T>
__device__ __host__
inline Quaternion<T> Quaternion<T>::operator/(T right) const
{
    return Quaternion<T>(vec / right);
}

template<class T>
__device__ __host__
inline void Quaternion<T>::operator*=(const Quaternion& right)
{
    Quaternion copy(*this);
    (*this) = copy * right;
}

template<class T>
__device__ __host__
inline void Quaternion<T>::operator*=(T right)
{
    vec *= right;
}

template<class T>
__device__ __host__
inline void Quaternion<T>::operator+=(const Quaternion& right)
{
    vec += right.vec;
}

template<class T>
__device__ __host__
inline void Quaternion<T>::operator-=(const Quaternion& right)
{
    vec -= right.vec;
}

template<class T>
__device__ __host__
inline void Quaternion<T>::operator/=(T right)
{
    vec /= right;
}

template<class T>
__device__ __host__
inline bool Quaternion<T>::operator==(const Quaternion& right) const
{
    return vec == right.vec;
}

template<class T>
__device__ __host__
inline bool Quaternion<T>::operator!=(const Quaternion& right) const
{
    return vec != right.vec;
}

template<class T>
__device__ __host__
inline Quaternion<T> Quaternion<T>::Normalize() const
{
    return Quaternion(vec.Normalize());
}

template<class T>
__device__ __host__
inline Quaternion<T>& Quaternion<T>::NormalizeSelf()
{
    vec.NormalizeSelf();
    return *this;
}

template<class T>
__device__ __host__
inline T Quaternion<T>::Length() const
{
    return vec.Length();
}

template<class T>
__device__ __host__
inline T Quaternion<T>::LengthSqr() const
{
    return vec.LengthSqr();
}

template<class T>
__device__ __host__
inline Quaternion<T> Quaternion<T>::Conjugate() const
{
    return Quaternion(vec[0], -vec[1], -vec[2], -vec[3]);
}

template<class T>
__device__ __host__
inline Quaternion<T>& Quaternion<T>::ConjugateSelf()
{
    vec[1] = -vec[1];
    vec[2] = -vec[2];
    vec[3] = -vec[3];
    return *this;
}

template<class T>
__device__ __host__
inline T Quaternion<T>::Dot(const Quaternion& right) const
{
    return vec.Dot(right.vec);
}

template<class T>
__device__ __host__
inline Vector<3, T> Quaternion<T>::ApplyRotation(const Vector<3, T>& vector) const
{
    // q * v * qInv
    // .Normalize();
    Quaternion qInv = Conjugate();
    Quaternion vectorQ(0.0f, vector[0], vector[1], vector[2]);

    Quaternion result((*this) * (vectorQ * qInv));
    return Vector<3,T>(result[1], result[2], result[3]);
}

template<class T>
__device__ __host__
inline Quaternion<T> Quat::NLerp(const Quaternion<T>& start, const Quaternion<T>& end, T t)
{
    return (start + t * (end - start));// .Normalize();
}

template<class T>
__device__ __host__
inline Quaternion<T> Quat::SLerp(const Quaternion<T>& start, const Quaternion<T>& end, T t)
{
    T cosTetha = start.Dot(end);
    // SLerp
    T angle = acos(cosTetha);
    return (start * sin(angle * (1.0f - t)) + end * sin(angle * t)) / sin(angle);
}

template<class T>
__device__ __host__
inline Quaternion<T> Quat::BarySLerp(const Quaternion<T>& q0,
                                     const Quaternion<T>& q1,
                                     const Quaternion<T>& q2,
                                     T a, T b)
{
    // Proper way to do this is
    // http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
    //
    // But it is computationally complex.
    //
    // However vertex quaternions of the triangle will be closer or same.
    // instead we can directly average them.
    // (for smooth edges neighbouring tri's face normal will be averaged)
    //
    // One thing to note is to check quaternions are close
    // and use conjugate in order to have proper average
    
    // Align tovards q0
    const Quaternion<T>& qA = q0;
    Quaternion<T> qB = (q1.Dot(q0) < 0) ? q1.Conjugate() : q1;
    Quaternion<T> qC = (q2.Dot(q0) < 0) ? q2.Conjugate() : q2;

    T c = (1 - a - b);
    Quaternion<T> result = qA * a + qB * b + qC * c;
    return result.Normalize();
}

template<class T>
__device__ __host__
inline Quaternion<T> Quat::RotationBetween(const Vector<3,T>& a, const Vector<3,T>& b)
{
    Vector<3, T> aCrossB = Cross(a, b);
    T aDotB = a.Dot(b);
    if(aCrossB != Vector<3, T>(static_cast<T>(0)))
        aCrossB.NormalizeSelf();
    return Quaternion<T>(acos(aDotB), aCrossB);
}

template<class T>
__device__ __host__
inline Quaternion<T> Quat::RotationBetweenZAxis(const Vector<3, T>& b)
{
    Vector<3, T> zCrossD(-b[1], b[0], 0);
    T zDotD = b[2];

    // Half angle teorem
    T sin = sqrt((1 - zDotD) * static_cast<T>(0.5));
    T cos = sqrt((zDotD + 1) * static_cast<T>(0.5));

    zCrossD.NormalizeSelf();
    T x = zCrossD[0] * sin;
    T y = zCrossD[1] * sin;
    T z = zCrossD[2] * sin;
    T w = cos;
    // Handle singularities
    if(abs(zDotD + 1) < MathConstants::Epsilon)
    {
        // Spaces are 180 degree apart
        // Define pi turn
        return Quaternion<T>(static_cast<T>(MathConstants::Pi_d),
                             Vector<3, T>(0, 1, 0));
    }
    else if(abs(zDotD - 1) < MathConstants::Epsilon)
    {
        // Spaces are nearly equavilent
        // Just turn identity
        return Quaternion<T>(1, 0, 0, 0);
    }
    else return Quaternion<T>(w, x, y, z);
}

template<class T>
__device__ __host__ 
inline Quaternion<T> operator*(T t, const Quaternion<T>& q)
{
    return q * t;
}

#include <algorithm>
#include <cmath>
#include "Log.h"

template <class T>
__device__ __host__
inline void TransformGen::Space(Quaternion<T>& q,
                                const Vector<3, T>& x,
                                const Vector<3, T>& y,
                                const Vector<3, T>& z)
{
    // Converting a Rotation Matrix to a Quaternion
    // Mike Day, Insomniac Games (2015)
    // https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

    // Coord Systems should match 
    // both should be right-handed coord system
    Vector3 crs = Cross(x, y);
    Vector3 diff = crs - z;
    //if((Cross(x, y) - z).Abs() >= Vector3(MathConstants::VeryLargeEpsilon))
    //    METU_ERROR_LOG("Quaternion: Coord Systam is not right handed.");
    assert((Cross(x, y) - z).Abs() <= Vector3(0.5));

    T t;
    if(z[2] < 0)
    {
        if(x[0] > y[1])
        {
            t = 1 + x[0] - y[1] - z[2];
            q = Quaternion<T>(y[2] - z[1],
                              t,
                              x[1] + y[0],
                              z[0] + x[2]);
        }
        else
        {
            t = 1 - x[0] + y[1] - z[2];
            q = Quaternion<T>(z[0] - x[2],
                              x[1] + y[0],
                              t, 
                              y[2] + z[1]);
        }
    }
    else
    {
        if(x[0] < -y[1])
        {
            t = 1 - x[0] - y[1] + z[2];
            q = Quaternion<T>(x[1] - y[0],
                              z[0] + x[2],
                              y[2] + z[1], 
                              t);
        }
        else
        {
            t = 1 + x[0] + y[1] + z[2];
            q = Quaternion<T>(t,
                              y[2] - z[1],
                              z[0] - x[2], 
                              x[1] - y[0]);
        }
    }
    q *= static_cast<T>(0.5) / sqrt(t);  
}

template <class T>
__device__ __host__
inline void TransformGen::InvSpace(Quaternion<T>& q,
                                   const Vector<3, T>& x,
                                   const Vector<3, T>& y,
                                   const Vector<3, T>& z)
{
    static constexpr Quaternion<T> Identity = Quaternion<T>(1, 0, 0, 0);
    using namespace std;
    T sqrtIn = max(static_cast<T>(0), 1 + x[0] - y[1] - z[2]);
    T qW = static_cast<T>(0.5) * sqrt(sqrtIn);
    if(qW <= MathConstants::Epsilon)
    {
        q = Identity;
        return;
    }

    T denom = static_cast<T>(0.25) / qW;
    T qX = (z[1] - y[2]) * denom;
    T qY = (x[2] - z[0]) * denom;
    T qZ = (y[0] - x[1]) * denom;

    q = Quaternion<T>(qW, qX, qY, qZ).ConjugateSelf();
}