#pragma once

/**

Arbitrary sized axis aligned bounding box.

N should be 2, 3 or 4 at most.

These are convenience register classes for GPU.

*/

#include "Vector.h"
#include <cfloat>

template<int N, class T, typename = ArithmeticEnable<T>>
class AABB;

template<int N, class T>
class alignas(ChooseVectorAlignment(N * sizeof(T))) AABB<N, T>
{
    public:
        static constexpr int            AABBVertexCount = 8;
        friend struct                   AABBOffsetChecker;

    private:
        Vector<N, T>                    min;
        Vector<N, T>                    max;



    protected:
    public:
        // Constructors & Destructor
        constexpr                                   AABB() = default;
        constexpr __device__ __host__               AABB(const Vector<N, T>& min,
                                                         const Vector<N, T>& max);
        __device__ __host__                         AABB(const T* dataMin,
                                                         const T* dataMax);

        template <class... Args0, class... Args1,
                  typename = AllArithmeticEnable<Args1...>,
                  typename = AllArithmeticEnable<Args0...>>
            constexpr __device__ __host__           AABB(const Args0... dataList0,
                                                         const Args1... dataList1);
                                                    ~AABB() = default;

        // Accessors
        __device__ __host__ const Vector<N, T>&     Min() const;
        __device__ __host__ const Vector<N, T>&     Max() const;
        __device__ __host__ Vector<N, T>            Min();
        __device__ __host__ Vector<N, T>            Max();

        // Mutators
        __device__ __host__ void                    SetMin(const Vector<N, T>&);
        __device__ __host__ void                    SetMax(const Vector<N, T>&);

        // Functionality
        __device__ __host__ Vector<N, T>            Span() const;
        __device__ __host__ Vector<N, T>            Centroid() const;
        __device__ __host__ AABB                    Union(const AABB&) const;
        __device__ __host__ AABB&                   UnionSelf(const AABB&);
        __device__ __host__ bool                    IsInside(const Vector<N, T>&) const;
        __device__ __host__ bool                    IsOutside(const Vector<N, T>&) const;
        __device__ __host__ Vector<N,T>             FurthestCorner(const Vector<N, T>&) const;

        // Intersection
        __device__ __host__ bool                    IntersectsSphere(const Vector3f& sphrPos,
                                                                     float sphrRadius);
};

// Typeless aabbs are defaulted to float
using AABB2 = AABB<2, float>;
using AABB3 = AABB<3, float>;
using AABB4 = AABB<4, float>;
// Float Type
using AABB2f = AABB<2, float>;
using AABB3f = AABB<3, float>;
using AABB4f = AABB<4, float>;
// Double Type
using AABB2d = AABB<2, double>;
using AABB3d = AABB<3, double>;
using AABB4d = AABB<4, double>;

// Requirements of Vectors
//static_assert(std::is_literal_type<AABB3>::value == true, "AABBs has to be literal types");
static_assert(std::is_trivially_copyable<AABB3>::value == true, "AABBs has to be trivially copyable");
static_assert(std::is_polymorphic<AABB3>::value == false, "AABBs should not be polymorphic");

struct AABBOffsetChecker
{
    // Some Sanity Traits
    static_assert(offsetof(AABB2f, min) == 0,
                  "AABB2f::min is not properly aligned for contagious mem read/write");
    static_assert(offsetof(AABB2f, max) == sizeof(Vector<2, float>),
                  "AABB2f:: max is not properly aligned for contagious mem read/write");
    static_assert(offsetof(AABB3f, min) == 0,
                  "AABB3f::min is not properly aligned for contagious mem read/write");
    static_assert(offsetof(AABB3f, max) == sizeof(Vector<3, float>),
                  "AABB3f:: max is not properly aligned for contagious mem read/write");
    static_assert(offsetof(AABB4f, min) == 0,
                  "AABB4f::min is not properly aligned for contagious mem read/write");
    static_assert(offsetof(AABB4f, max) == sizeof(Vector<4, float>),
                  "AABB4f:: max is not properly aligned for contagious mem read/write");
    //
    static_assert(offsetof(AABB2d, min) == 0,
                  "AABB2d::min is not properly aligned for contagious mem read/write");
    static_assert(offsetof(AABB2d, max) == sizeof(Vector<2, double>),
                  "AABB2d:: max is not properly aligned for contagious mem read/write");
    static_assert(offsetof(AABB3d, min) == 0,
                  "AABB3d::min is not properly aligned for contagious mem read/write");
    static_assert(offsetof(AABB3d, max) == sizeof(Vector<3, double>),
                  "AABB3d:: max is not properly aligned for contagious mem read/write");
    static_assert(offsetof(AABB4d, min) == 0,
                  "AABB4d::min is not properly aligned for contagious mem read/write");
    static_assert(offsetof(AABB4d, max) == sizeof(Vector<4, double>),
                  "AABB4d:: max is not properly aligned for contagious mem read/write");
};

// Implementation
#include "AABB.hpp" // CPU & GPU

// Zero Constants
static constexpr AABB2f ZeroAABB2f = AABB2f(Vector2f(0.0f, 0.0f),
                                            Vector2f(0.0f, 0.0f));
static constexpr AABB3f ZeroAABB3f = AABB3f(Vector3f(0.0f, 0.0f, 0.0f),
                                            Vector3f(0.0f, 0.0f, 0.0f));
static constexpr AABB4f ZeroAABB4f = AABB4f(Vector4f(0.0f, 0.0f, 0.0f, 0.0f),
                                            Vector4f(0.0f, 0.0f, 0.0f, 0.0f));

static constexpr AABB2d ZeroAABB2d = AABB2d(Vector2d(0.0, 0.0), Vector2d(0.0, 0.0));
static constexpr AABB3d ZeroAABB3d = AABB3d(Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0));
static constexpr AABB4d ZeroAABB4d = AABB4d(Vector4d(0.0, 0.0, 0.0, 0.0),
                                            Vector4d(0.0, 0.0, 0.0, 0.0));

static constexpr AABB2f CoveringAABB2f = AABB2f(Vector2f(-FLT_MAX, -FLT_MAX),
                                                Vector2f(FLT_MAX, FLT_MAX));
static constexpr AABB3f CoveringAABB3f = AABB3f(Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX),
                                                Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));
static constexpr AABB4f CoveringAABB4f = AABB4f(Vector4f(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX),
                                                Vector4f(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX));

static constexpr AABB2d CoveringAABB2d = AABB2d(Vector2d(-DBL_MAX, -DBL_MAX),
                                                Vector2d(DBL_MAX, DBL_MAX));
static constexpr AABB3d CoveringAABB3d = AABB3d(Vector3d(-DBL_MAX, -DBL_MAX, -DBL_MAX),
                                                Vector3d(DBL_MAX, DBL_MAX, DBL_MAX));
static constexpr AABB4d CoveringAABB4d = AABB4d(Vector4d(-DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX),
                                                Vector4d(DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX));

static constexpr AABB2f NegativeAABB2f = AABB2f(Vector2f(FLT_MAX, FLT_MAX),
                                                Vector2f(-FLT_MAX, -FLT_MAX));
static constexpr AABB3f NegativeAABB3f = AABB3f(Vector3f(FLT_MAX, FLT_MAX, FLT_MAX),
                                                Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
static constexpr AABB4f NegativeAABB4f = AABB4f(Vector4f(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX),
                                                Vector4f(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX));

static constexpr AABB2d NegativeAABB2d = AABB2d(Vector2d(DBL_MAX, DBL_MAX),
                                                Vector2d(-DBL_MAX, -DBL_MAX));
static constexpr AABB3d NegativeAABB3d = AABB3d(Vector3d(DBL_MAX, DBL_MAX, DBL_MAX),
                                                Vector3d(-DBL_MAX, -DBL_MAX, -DBL_MAX));
static constexpr AABB4d NegativeAABB4d = AABB4d(Vector4d(DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX),
                                                Vector4d(-DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX));

static constexpr AABB2 CoveringAABB2 = CoveringAABB2f;
static constexpr AABB3 CoveringAABB3 = CoveringAABB3f;
static constexpr AABB4 CoveringAABB4 = CoveringAABB4f;

static constexpr AABB2 NegativeAABB2 = NegativeAABB2f;
static constexpr AABB3 NegativeAABB3 = NegativeAABB3f;
static constexpr AABB4 NegativeAABB4 = NegativeAABB4f;

static constexpr AABB2 ZeroAABB2 = ZeroAABB2f;
static constexpr AABB3 ZeroAABB3 = ZeroAABB3f;
static constexpr AABB4 ZeroAABB4 = ZeroAABB4f;

//// AABB Extern
//extern template class AABB<2, float>;
//extern template class AABB<3, float>;
//extern template class AABB<4, float>;
//
//extern template class AABB<2, double>;
//extern template class AABB<3, double>;
//extern template class AABB<4, double>;
