#pragma once

/**

Arbitrary sized square matrix.

N should be 2, 3, 4 at most.

*/

#include <algorithm>
#include <type_traits>
#include "CudaCheck.h"
#include "Vector.h"
#include "Quaternion.h"
#include "Unreachable.h"

template<int N, class T, class... Args>
using AllVectorEnable = typename std::enable_if<std::conjunction<std::is_same<Vector<N, T>, Args>...>::value>::type;

template<int N, class T, typename = ArithmeticEnable<T>>
class Matrix;

template<int N, class T>
class alignas(ChooseVectorAlignment(N * sizeof(T))) Matrix<N, T>
{
    static_assert(N == 2 || N == 3 || N == 4, "Matrix size should be 2x2, 3x3 or 4x4");

    private:
        T                                   matrix[N * N];

    protected:
    public:
        // Constructors & Destructor
        constexpr                           Matrix() = default;
        template <class C, typename = ArithmeticEnable<C>>
        __device__ __host__                 Matrix(C);
        template <class C, typename = ArithmeticEnable<C>>
        __device__ __host__                 Matrix(const C* data);
        template <class... Args, typename = AllArithmeticEnable<Args...>>
        constexpr __device__ __host__       Matrix(const Args... dataList);
        __device__ __host__                 Matrix(const Vector<N, T> columns[]);
        template <int M, typename = std::enable_if_t<(M >= N)>>
        __device__ __host__                 Matrix(const Matrix<M, T>&);
                                            ~Matrix() = default;

        // MVC bug? these trigger std::trivially_copyable static assert
        //                                  Matrix(const Matrix&) = default;
        //Matrix&                           operator=(const Matrix&) = default;

        // Accessors
        __device__ __host__ explicit        operator T* ();
        __device__ __host__ explicit        operator const T* () const;
        __device__ __host__ T&              operator[](int);
        __device__ __host__ const T&        operator[](int) const;
        __device__ __host__ T&              operator()(int row, int column);
        __device__ __host__ const T&        operator()(int row, int column) const;

        // Modify
        __device__ __host__ void            operator+=(const Matrix&);
        __device__ __host__ void            operator-=(const Matrix&);
        __device__ __host__ void            operator*=(const Matrix&);
        __device__ __host__ void            operator*=(T);
        __device__ __host__ void            operator/=(const Matrix&);
        __device__ __host__ void            operator/=(T);

        __device__ __host__ Matrix                      operator+(const Matrix&) const;
        __device__ __host__ Matrix                      operator-(const Matrix&) const;
        template<class Q = T>
        __device__ __host__ SignedEnable<Q, Matrix>     operator-() const;
        __device__ __host__ Matrix                      operator/(const Matrix&) const;
        __device__ __host__ Matrix                      operator/(T) const;

        __device__ __host__ Matrix                      operator*(const Matrix&) const;
        template<int M>
        __device__ __host__ Vector<M, T>                operator*(const Vector<M, T>&) const;
        __device__ __host__ Matrix                      operator*(T) const;

        // Logic
        __device__ __host__ bool                        operator==(const Matrix&) const;
        __device__ __host__ bool                        operator!=(const Matrix&) const;

        // Utility
        __device__ __host__ T                           Determinant() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Matrix>      Inverse() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Matrix&>     InverseSelf();
        __device__ __host__ Matrix                      Transpose() const;
        __device__ __host__ Matrix&                     TransposeSelf();

        __device__ __host__ Matrix          Clamp(const Matrix&, const Matrix&) const;
        __device__ __host__ Matrix          Clamp(T min, T max) const;
        __device__ __host__ Matrix&         ClampSelf(const Matrix&, const Matrix&);
        __device__ __host__ Matrix&         ClampSelf(T min, T max);

        template<class Q = T>
        __device__ __host__ SignedEnable<Q, Matrix>     Abs() const;
        template<class Q = T>
        __device__ __host__ SignedEnable<Q, Matrix&>    AbsSelf();
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Matrix>      Round() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Matrix&>     RoundSelf();
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Matrix>      Floor() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Matrix&>     FloorSelf();
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Matrix>      Ceil() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Matrix&>     CeilSelf();

        template<class Q = T>
        static __device__ __host__ FloatEnable<Q, Matrix>   Lerp(const Matrix&, const Matrix&, T);

        static __device__ __host__ Matrix       Min(const Matrix&, const Matrix&);
        static __device__ __host__ Matrix       Min(const Matrix&, T);
        static __device__ __host__ Matrix       Max(const Matrix&, const Matrix&);
        static __device__ __host__ Matrix       Max(const Matrix&, T);
};

// Determinants
template<class T>
__device__ __host__ T Determinant2(const T*);

template<class T>
__device__ __host__ T Determinant3(const T*);

template<class T>
__device__ __host__ T Determinant4(const T*);

// Inverse
template<class T>
__device__ __host__ Matrix<2, T> Inverse2(const T*);

template<class T>
__device__ __host__ Matrix<3, T> Inverse3(const T*);

template<class T>
__device__ __host__ Matrix<4, T> Inverse4(const T*);

// Left Scalar operators
template<int N, class T>
__device__ __host__ Matrix<N, T> operator*(float, const Matrix<N, T>&);

// Typeless matrices are defaulted to float
using Matrix2x2 = Matrix<2, float>;
using Matrix3x3 = Matrix<3, float>;
using Matrix4x4 = Matrix<4, float>;
// Float Type
using Matrix2x2f = Matrix<2, float>;
using Matrix3x3f = Matrix<3, float>;
using Matrix4x4f = Matrix<4, float>;
// Double Type
using Matrix2x2d = Matrix<2, double>;
using Matrix3x3d = Matrix<3, double>;
using Matrix4x4d = Matrix<4, double>;
// Integer Type
using Matrix2x2i = Matrix<2, int>;
using Matrix3x3i = Matrix<3, int>;
using Matrix4x4i = Matrix<4, int>;
// Unsigned Integer Type
using Matrix2x2ui = Matrix<2, unsigned int>;
using Matrix3x3ui = Matrix<3, unsigned int>;
using Matrix4x4ui = Matrix<4, unsigned int>;

// Requirements of Vectors
//static_assert(std::is_literal_type<Matrix3x3>::value == true, "Matrices has to be literal types");
static_assert(std::is_trivially_copyable<Matrix3x3>::value == true, "Matrices has to be trivially copyable");
static_assert(std::is_polymorphic<Matrix3x3>::value == false, "Matrices should not be polymorphic");

//// Special 4x4 Matrix Operation
//template<class T>
//static __device__ __host__ Vector<3, T> ExtractScaleInfo(const Matrix<4, T>&);

// Spacial Matrix4x4 -> Matrix3x3
template<class T>
static __device__ __host__ Matrix<4, T> ToMatrix4x4(const Matrix<3, T>&);

// Transformation Matrix Generation
namespace TransformGen
{
    // Extraction Functions
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Vector<3, T>     ExtractScale(const Matrix<4, T>&);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Vector<3, T>     ExtractTranslation(const Matrix<4, T>&);

    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Matrix<4, T>     Translate(const Vector<3, T>&);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Matrix<4, T>     Scale(T);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Matrix <4, T>    Scale(T x, T y, T z);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Matrix<4, T>     Rotate(T angle, const Vector<3, T>&);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Matrix<4, T>     Rotate(const Quaternion<T>&);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Matrix<4, T>     Perspective(T fovXRadians, T aspectRatio,
                                 T nearPlane, T farPlane);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Matrix<4, T>     Ortogonal(T left, T right,
                               T top, T bottom,
                               T nearPlane, T farPlane);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Matrix<4, T>     Ortogonal(T width, T height,
                               T nearPlane, T farPlane);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    Matrix<4, T>     LookAt(const Vector<3, T>& eyePos,
                            const Vector<3, T>& at,
                            const Vector<3, T>& up);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    void             Space(Matrix<3, T>&,
                           const Vector<3, T>& x,
                           const Vector<3, T>& y,
                           const Vector<3, T>& z);
    template<class T, typename = FloatEnable<T>>
    __device__ __host__ HYBRID_INLINE
    void             InvSpace(Matrix<3, T>&,
                              const Vector<3, T>& x,
                              const Vector<3, T>& y,
                              const Vector<3, T>& z);
}

// Implementation
#include "Matrix.hpp"   // CPU & GPU

// Constants
static constexpr Matrix2x2  Indentity2x2 = Matrix2x2(1.0f, 0.0f,
                                                     0.0f, 1.0f);
static constexpr Matrix3x3  Indentity3x3 = Matrix3x3(1.0f, 0.0f, 0.0f,
                                                     0.0f, 1.0f, 0.0f,
                                                     0.0f, 0.0f, 1.0f);
static constexpr Matrix4x4  Indentity4x4 = Matrix4x4(1.0f, 0.0f, 0.0f, 0.0f,
                                                     0.0f, 1.0f, 0.0f, 0.0f,
                                                     0.0f, 0.0f, 1.0f, 0.0f,
                                                     0.0f, 0.0f, 0.0f, 1.0f);

// Zeros
static constexpr Matrix2x2f  Zero2x2f = Matrix2x2f(0.0f, 0.0f,
                                                   0.0f, 0.0f);
static constexpr Matrix3x3f  Zero3x3f = Matrix3x3f(0.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f);
static constexpr Matrix4x4f Zero4x4f = Matrix4x4f(0.0f, 0.0f, 0.0f, 0.0f,
                                                  0.0f, 0.0f, 0.0f, 0.0f,
                                                  0.0f, 0.0f, 0.0f, 0.0f,
                                                  0.0f, 0.0f, 0.0f, 0.0f);

static constexpr Matrix2x2d  Zero2x2d = Matrix2x2d(0.0f, 0.0f,
                                                   0.0f, 0.0f);
static constexpr Matrix3x3d  Zero3x3d = Matrix3x3d(0.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f);
static constexpr Matrix4x4d  Zero4x4d = Matrix4x4d(0.0f, 0.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f, 0.0f);

static constexpr Matrix2x2i  Zero2x2i = Matrix2x2i(0, 0,
                                                   0, 0);
static constexpr Matrix3x3i  Zero3x3i = Matrix3x3i(0, 0, 0,
                                                   0, 0, 0,
                                                   0, 0, 0);
static constexpr Matrix4x4i  Zero4x4i = Matrix4x4i(0, 0, 0, 0,
                                                   0, 0, 0, 0,
                                                   0, 0, 0, 0,
                                                   0, 0, 0, 0);

static constexpr Matrix2x2ui  Zero2x2ui = Matrix2x2ui(0u, 0u,
                                                      0u, 0u);
static constexpr Matrix3x3ui  Zero3x3ui = Matrix3x3ui(0u, 0u, 0u,
                                                      0u, 0u, 0u,
                                                      0u, 0u, 0u);
static constexpr Matrix4x4ui  Zero4x4ui = Matrix4x4ui(0u, 0u, 0u, 0u,
                                                      0u, 0u, 0u, 0u,
                                                      0u, 0u, 0u, 0u,
                                                      0u, 0u, 0u, 0u);

static constexpr Matrix2x2  Zero2x2 = Zero2x2f;
static constexpr Matrix3x3  Zero3x3 = Zero3x3f;
static constexpr Matrix4x4  Zero4x4 = Zero4x4f;

// Matrix Traits
template<class T>
struct IsMatrixType
{
    static constexpr bool value =
        std::is_same<T, Matrix2x2f>::value  ||
        std::is_same<T, Matrix2x2d>::value  ||
        std::is_same<T, Matrix2x2i>::value  ||
        std::is_same<T, Matrix2x2ui>::value ||
        std::is_same<T, Matrix3x3f>::value  ||
        std::is_same<T, Matrix3x3d>::value  ||
        std::is_same<T, Matrix3x3i>::value  ||
        std::is_same<T, Matrix3x3ui>::value ||
        std::is_same<T, Matrix4x4f>::value  ||
        std::is_same<T, Matrix4x4d>::value  ||
        std::is_same<T, Matrix4x4i>::value  ||
        std::is_same<T, Matrix4x4ui>::value;
};

// // Matrix Extern
// extern template class Matrix<2, float>;
// extern template class Matrix<2, double>;
// extern template class Matrix<2, int>;
// extern template class Matrix<2, unsigned int>;

// extern template class Matrix<3, float>;
// extern template class Matrix<3, double>;
// extern template class Matrix<3, int>;
// extern template class Matrix<3, unsigned int>;

// extern template class Matrix<4, float>;
// extern template class Matrix<4, double>;
// extern template class Matrix<4, int>;
// extern template class Matrix<4, unsigned int>;
