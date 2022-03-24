#pragma once

/**

Arbitrary sized vector. Vector is column vector (N x 1 matrix)
which means that it can only be multiplied with matrices from right.

N should be 2, 3 or 4 at most.

*/

#include <cmath>
#include <type_traits>
#include <tuple>
#include "CudaCheck.h"

#ifdef METU_CUDA
    #include <cuda_fp16.h>
#endif

template <class T>
struct IsArithmeticType : std::is_arithmetic<T> {};

template <class T>
struct IsFloatType : std::is_floating_point<T> {};

template <class T>
struct IsSignedType : std::is_signed<T> {};

#ifdef METU_CUDA
    template <>
    struct IsArithmeticType<half> : std::true_type {};

    template <>
    struct IsFloatType<half> : std::true_type {};

    template <>
    struct IsSignedType<half> : std::true_type {};
#endif

template<class T>
using ArithmeticEnable = typename std::enable_if<IsArithmeticType<T>::value>::type;

template<class... Args>
using AllArithmeticEnable = typename std::enable_if<std::conjunction<IsArithmeticType<Args>...>::value>::type;

template<class T, class RType = void>
using FloatEnable = typename std::enable_if<IsFloatType<T>::value, RType>::type;

template<class T, class RType = void>
using IntegralEnable = typename std::enable_if<std::is_integral<T>::value, RType>::type;

template<class T, class RType = void>
using SignedEnable = typename std::enable_if<IsSignedType<T>::value, RType>::type;

template<int N, class T, typename = ArithmeticEnable<T>>
class Vector;

static constexpr size_t ChooseVectorAlignment(size_t totalSize)
{
    if(totalSize <= 2)
        return 2;
    else if(totalSize <= 4)
        return 4;           // 1byte Vector Types
    else if(totalSize <= 8)
        return 8;           // 4byte Vector2 Types
    else if(totalSize < 16)
        return 4;           // 4byte Vector3 Types
    else
        return 16;          // 4byte Vector4 Types
}

template<int N, class T>
class alignas(ChooseVectorAlignment(N * sizeof(T))) Vector<N, T>
{
    static_assert(N == 2 || N == 3 || N == 4, "Vector size should be 2, 3 or 4");

    private:
        T                                       vector[N];

    protected:
    public:
        // Constructors & Destructor
        constexpr                               Vector() = default;
        template<class C, typename = ArithmeticEnable<C>>
        __device__ __host__ explicit            Vector(C);
        template<class C, typename = ArithmeticEnable<C>>
        __device__ __host__ explicit            Vector(const C* data);
        template <class... Args, typename = AllArithmeticEnable<Args...>>
        constexpr __device__ __host__ explicit  Vector(const Args... dataList);
        template <class... Args, typename = std::enable_if_t<((N - sizeof...(Args)) > 1)>>
        __device__ __host__                     Vector(const Vector<N - sizeof...(Args), T>&,
                                                       const Args... dataList);
        template <int M, typename = std::enable_if_t<(M >= N)>>
        __device__ __host__                     Vector(const Vector<M, T>&);
                                                ~Vector() = default;

        // MVC bug? these trigger std::trivially_copyable static assert
        // __device__ __host__              Vector(const Vector&) = default;
        // __device__ __host__ Vector&      operator=(const Vector&) = default;

        // Accessors
        __device__ __host__ explicit            operator T* ();
        __device__ __host__ explicit            operator const T* () const;
        __device__ __host__ T&                  operator[](int);
        __device__ __host__ constexpr const T&  operator[](int) const;

        // Type cast
        template<int M, class C, typename = std::enable_if_t<(M <= N)>>
        __device__ __host__ explicit                    operator Vector<M, C>() const;

        // Modify
        __device__ __host__ void                        operator+=(const Vector&);
        __device__ __host__ void                        operator-=(const Vector&);
        __device__ __host__ void                        operator*=(const Vector&);
        __device__ __host__ void                        operator*=(T);
        __device__ __host__ void                        operator/=(const Vector&);
        __device__ __host__ void                        operator/=(T);

        __device__ __host__ Vector                      operator+(const Vector&) const;
        __device__ __host__ Vector                      operator+(T) const;
        __device__ __host__ Vector                      operator-(const Vector&) const;
        __device__ __host__ Vector                      operator-(T) const;
        template<class Q = T>
        __device__ __host__ SignedEnable<Q, Vector>     operator-() const;
        __device__ __host__ Vector                      operator*(const Vector&) const;
        __device__ __host__ Vector                      operator*(T) const;
        __device__ __host__ Vector                      operator/(const Vector&) const;
        __device__ __host__ Vector                      operator/(T) const;

        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      operator%(const Vector&) const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      operator%(T) const;
        template<class Q = T>
        __device__ __host__ IntegralEnable<Q, Vector>   operator%(const Vector&) const;
        template<class Q = T>
        __device__ __host__ IntegralEnable<Q, Vector>   operator%(T) const;

        // Logic
        __device__ __host__ bool                        operator==(const Vector&) const;
        __device__ __host__ bool                        operator!=(const Vector&) const;
        __device__ __host__ bool                        operator<(const Vector&) const;
        __device__ __host__ bool                        operator<=(const Vector&) const;
        __device__ __host__ bool                        operator>(const Vector&) const;
        __device__ __host__ bool                        operator>=(const Vector&) const;

        // Utility
        __device__ __host__ T                           Dot(const Vector&) const;

        // Reduction
        __device__ __host__ T                           Sum() const;
        __device__ __host__ T                           Multiply() const;
        // Max Min Reduction functions are selections instead
        // since it sometimes useful to fetch the which index
        // (axis) is maximum so that you can do other stuff wrt. it.
        __device__ __host__ int                         Max() const;
        __device__ __host__ int                         Min() const;

        template<class Q = T>
        __device__ __host__ FloatEnable<Q, T>           Length() const;
        __device__ __host__ T                           LengthSqr() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      Normalize() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector&>     NormalizeSelf();
        __device__ __host__ Vector                      Clamp(const Vector&, const Vector&) const;
        __device__ __host__ Vector                      Clamp(T min, T max) const;
        __device__ __host__ Vector&                     ClampSelf(const Vector&, const Vector&);
        __device__ __host__ Vector&                     ClampSelf(T min, T max);
        __device__ __host__ bool                        HasNaN() const;

        template<class Q = T>
        __device__ __host__ SignedEnable<Q, Vector>     Abs() const;
        template<class Q = T>
        __device__ __host__ SignedEnable<Q, Vector&>    AbsSelf();
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      Round() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector&>     RoundSelf();
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      Floor() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector&>     FloorSelf();
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      Ceil() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector&>     CeilSelf();

        static __device__ __host__ Vector               Min(const Vector&, const Vector&);
        static __device__ __host__ Vector               Min(const Vector&, T);
        static __device__ __host__ Vector               Max(const Vector&, const Vector&);
        static __device__ __host__ Vector               Max(const Vector&, T);

        template<class Q = T>
        static __device__ __host__ FloatEnable<Q, Vector>   Lerp(const Vector&,
                                                                 const Vector&,
                                                                 T);
};

// Left scalars
template<int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> operator*(T, const Vector<N, T>&);

// Typeless vectors are defaulted to float
using Vector2 = Vector<2, float>;
using Vector3 = Vector<3, float>;
using Vector4 = Vector<4, float>;
// Float Type
using Vector2f = Vector<2, float>;
using Vector3f = Vector<3, float>;
using Vector4f = Vector<4, float>;
// Double Type
using Vector2d = Vector<2, double>;
using Vector3d = Vector<3, double>;
using Vector4d = Vector<4, double>;
// Integer Type
using Vector2i = Vector<2, int32_t>;
using Vector3i = Vector<3, int32_t>;
using Vector4i = Vector<4, int32_t>;
// Unsigned Integer Type
using Vector2ui = Vector<2, uint32_t>;
using Vector3ui = Vector<3, uint32_t>;
using Vector4ui = Vector<4, uint32_t>;
// Long Types
using Vector2l  = Vector<2, int64_t>;
using Vector2ul = Vector<2, uint64_t>;
using Vector3ul = Vector<3, uint64_t>;
// Short
using Vector2s   = Vector<2, int16_t>;
using Vector2us  = Vector<2, uint16_t>;
using Vector3s   = Vector<3, int16_t>;
using Vector3us  = Vector<3, uint16_t>;
using Vector4s   = Vector<4, int16_t>;
using Vector4us  = Vector<4, uint16_t>;
// Byte
using Vector2c   = Vector<2, int8_t>;
using Vector2uc  = Vector<2, uint8_t>;
using Vector3c   = Vector<3, int8_t>;
using Vector3uc  = Vector<3, uint8_t>;
using Vector4c   = Vector<4, int8_t>;
using Vector4uc  = Vector<4, uint8_t>;

#ifdef METU_CUDA
    using Vector2h = Vector<2, half>;
    using Vector3h = Vector<3, half>;
    using Vector4h = Vector<4, half>;
#endif

// Requirements of Vectors
//static_assert(std::is_literal_type<Vector3>::value == true, "Vectors has to be literal types");
static_assert(std::is_trivially_copyable<Vector3>::value == true, "Vectors has to be trivially copyable");
static_assert(std::is_polymorphic<Vector3>::value == false, "Vectors should not be polymorphic");

// Alignment Checks
static_assert(sizeof(Vector2) == 8, "Vector2 should be tightly packed");
static_assert(sizeof(Vector3) == 12, "Vector3 should be tightly packed");
static_assert(sizeof(Vector4) == 16, "Vector4 should be tightly packed");

// Cross product (only for 3d vectors)
template <class T>
__device__ __host__ HYBRID_INLINE
Vector<3, T> Cross(const Vector<3, T>&, const Vector<3, T>&);

// Arbitrary Orthogonal Vector Generation (only for 3D Vectors)
template <class T>
__device__ __host__ HYBRID_INLINE
Vector<3, T> OrthogonalVector(const Vector<3, T>&);

// Implementation
#include "Vector.hpp"   // CPU & GPU

// Basic Constants
static constexpr Vector3 XAxis = Vector3(1.0f, 0.0f, 0.0f);
static constexpr Vector3 YAxis = Vector3(0.0f, 1.0f, 0.0f);
static constexpr Vector3 ZAxis = Vector3(0.0f, 0.0f, 1.0f);

// Zero Constants
static constexpr Vector2 Zero2f = Vector2(0.0f, 0.0f);
static constexpr Vector3 Zero3f = Vector3(0.0f, 0.0f, 0.0f);
static constexpr Vector4 Zero4f = Vector4(0.0f, 0.0f, 0.0f, 0.0f);

static constexpr Vector2d Zero2d = Vector2d(0.0, 0.0);
static constexpr Vector3d Zero3d = Vector3d(0.0, 0.0, 0.0);
static constexpr Vector4d Zero4d = Vector4d(0.0, 0.0, 0.0, 0.0);

static constexpr Vector2i Zero2i = Vector2i(0, 0);
static constexpr Vector3i Zero3i = Vector3i(0, 0, 0);
static constexpr Vector4i Zero4i = Vector4i(0, 0, 0, 0);

static constexpr Vector2ui Zero2ui = Vector2ui(0u, 0u);
static constexpr Vector3ui Zero3ui = Vector3ui(0u, 0u, 0u);
static constexpr Vector4ui Zero4ui = Vector4ui(0u, 0u, 0u, 0u);

static constexpr Vector2ul Zero2ul = Vector2ul(0ul, 0ul);

static constexpr Vector2 Zero2 = Zero2f;
static constexpr Vector3 Zero3 = Zero3f;
static constexpr Vector4 Zero4 = Zero4f;

// No constepxr support of half on cuda
//#ifdef METU_CUDA
//    static constexpr Vector2h Zero2h = Vector2h(static_cast<half>(0.0f),
//                                                static_cast<half>(0.0f));
//    static constexpr Vector3h Zero3h = Vector3h(0.0f, 0.0f, 0.0f);
//    static constexpr Vector4h Zero4h = Vector4h(0.0f, 0.0f, 0.0f, 0.0f);
//#endif

// Vector Traits
template<class T>
struct IsVectorType
{
    static constexpr bool value =
        std::is_same<T, Vector2f>::value ||
        std::is_same<T, Vector2d>::value ||
        std::is_same<T, Vector2i>::value ||
        std::is_same<T, Vector2ui>::value ||
        std::is_same<T, Vector2l>::value ||
        std::is_same<T, Vector2ul>::value ||
        std::is_same<T, Vector2s>::value ||
        std::is_same<T, Vector2us>::value ||

        std::is_same<T, Vector3f>::value ||
        std::is_same<T, Vector3d>::value ||
        std::is_same<T, Vector3i>::value ||
        std::is_same<T, Vector3ui>::value ||

        std::is_same<T, Vector4f>::value ||
        std::is_same<T, Vector4d>::value ||
        std::is_same<T, Vector4i>::value ||
        std::is_same<T, Vector4ui>::value ||
        std::is_same<T, Vector4s>::value ||
        std::is_same<T, Vector4us>::value;
};

// // Vector Extern
// // Float Type
// extern template class Vector<2, float>;
// extern template class Vector<3, float>;
// extern template class Vector<4, float>;
// // Double Type
// extern template class Vector<2, double>;
// extern template class Vector<3, double>;
// extern template class Vector<4, double>;
// // Integer Type
// extern template class Vector<2, int32_t>;
// extern template class Vector<3, int32_t>;
// extern template class Vector<4, int32_t>;
// // Unsigned Integer Type
// // IMPORTANT: This (and the non extern brother in the Vector.cu)
// // Breaks the at the __cudaRegisterFatbinary() when you load VisorGL.so.
// // I don't know why...
// // extern template class Vector<2, uint32_t>;
// // extern template class Vector<3, uint32_t>;
// // extern template class Vector<4, uint32_t>;
// // Long Types
// extern template class Vector<2, int64_t>;
// extern template class Vector<2, uint64_t>;
// // Short Type
// extern template class Vector<2, int16_t>;
// extern template class Vector<3, int16_t>;
// extern template class Vector<4, int16_t>;
// // Unsigned Short Type
// extern template class Vector<2, uint16_t>;
// extern template class Vector<3, uint16_t>;
// extern template class Vector<4, uint16_t>;
// // Byte Type
// extern template class Vector<2, int8_t>;
// extern template class Vector<3, int8_t>;
// extern template class Vector<4, int8_t>;
// // Unsigned Byte Type
// extern template class Vector<2, uint8_t>;
// extern template class Vector<3, uint8_t>;
// extern template class Vector<4, uint8_t>;