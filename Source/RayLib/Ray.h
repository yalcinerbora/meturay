#pragma once
/**

Ray struct for convenient usability.

*/
#include "Matrix.h"
#include "Vector.h"
#include "Quaternion.h"
#include "HybridFunctions.h"

#include <limits>
#include <cfloat>

template<class T, typename = ArithmeticEnable<T>>
class Ray;
template<class T>
class Ray<T>
{
    private:
    Vector<3,T>                         direction;
    Vector<3,T>                         position;

    protected:
    public:
    // Constructors & Destructor
    constexpr                           Ray() = default;
    constexpr HYBRD_FUNC                Ray(const Vector<3,T>& direction,
                                            const Vector<3,T>& position);
    constexpr HYBRD_FUNC                Ray(const Vector<3,T>[2]);
                                        Ray(const Ray&) = default;
                                        ~Ray() = default;
    Ray&                                operator=(const Ray&) = default;

    // Assignment Operators
    HYBRD_FUNC Ray&                     operator=(const Vector<3, T>[2]);

    HYBRD_FUNC const Vector<3,T>&       getDirection() const;
    HYBRD_FUNC const Vector<3,T>&       getPosition() const;

    // Intersections
    HYBRD_FUNC bool         IntersectsSphere(Vector<3, T>& pos, T& t,
                                                const Vector<3, T>& sphereCenter,
                                                T sphereRadius) const;
    HYBRD_FUNC bool         IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                                const Vector<3, T> triCorners[3],
                                                bool cullFace = true) const;
    HYBRD_FUNC bool         IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                                const Vector<3, T>& t0,
                                                const Vector<3, T>& t1,
                                                const Vector<3, T>& t2,
                                                bool cullFace = true) const;
    HYBRD_FUNC bool         IntersectsPlane(Vector<3, T>& position, T& t,
                                            const Vector<3, T>& planePos,
                                            const Vector<3, T>& normal);
    HYBRD_FUNC bool         IntersectsAABB(const Vector<3, T>& min,
                                            const Vector<3, T>& max,
                                            const Vector<2, T>& tMinMax = Vector<2, T>(-INFINITY, INFINITY)) const;
    HYBRD_FUNC bool         IntersectsAABB(Vector<3,T>& pos, T& t,
                                            const Vector<3, T>& min,
                                            const Vector<3, T>& max,
                                            const Vector<2, T>& tMinMax = Vector<2, T>(-INFINITY, INFINITY)) const;

    // Utility
    HYBRD_FUNC [[nodiscard]] Ray    Reflect(const Vector<3, T>& normal) const;
    HYBRD_FUNC Ray&                 ReflectSelf(const Vector<3, T>& normal);
    HYBRD_FUNC bool                 Refract(Ray& out, const Vector<3, T>& normal,
                                            T fromMedium, T toMedium) const;
    HYBRD_FUNC bool                 RefractSelf(const Vector<3, T>& normal,
                                                T fromMedium, T toMedium);

    // Randomization (Hemispherical)
    HYBRD_FUNC static Ray           RandomRayCosine(T xi0, T xi1,
                                                    const Vector<3, T>& normal,
                                                    const Vector<3, T>& position);
    HYBRD_FUNC static Ray           RandomRayUnfirom(T xi0, T xi1,
                                                     const Vector<3, T>& normal,
                                                     const Vector<3, T>& position);

    HYBRD_FUNC [[nodiscard]] Ray            NormalizeDir() const;
    HYBRD_FUNC Ray&                         NormalizeDirSelf();
    HYBRD_FUNC [[nodiscard]] Ray            Advance(T) const;
    HYBRD_FUNC [[nodiscard]] Ray            Advance(T t, const Vector<3,T>& direction) const;
    HYBRD_FUNC Ray&                         AdvanceSelf(T);
    HYBRD_FUNC Ray&                         AdvanceSelf(T t, const Vector<3, T>& direction);
    HYBRD_FUNC [[nodiscard]] Ray            Transform(const Quaternion<T>&) const;
    HYBRD_FUNC [[nodiscard]] Ray            Transform(const Matrix<3, T>&) const;
    HYBRD_FUNC [[nodiscard]] Ray            Transform(const Matrix<4, T>&) const;
    HYBRD_FUNC Ray                          TransformSelf(const Quaternion<T>&);
    HYBRD_FUNC Ray&                         TransformSelf(const Matrix<3, T>&);
    HYBRD_FUNC Ray&                         TransformSelf(const Matrix<4, T>&);
    HYBRD_FUNC [[nodiscard]] Vector<3,T>    AdvancedPos(T t) const;
    HYBRD_FUNC [[nodiscard]] Ray            Nudge(const Vector<3, T>& direction, T curvatureOffset = T(0)) const;
    HYBRD_FUNC Ray&                         NudgeSelf(const Vector<3, T>& direction, T curvatureOffset = T(0));
};

using RayF = Ray<float>;
using RayD = Ray<double>;

// Requirements of IERay
static_assert(std::is_trivially_copyable<RayF>::value == true, "Ray has to be trivially copyable");
static_assert(std::is_polymorphic<RayF>::value == false, "Ray should not be polymorphic");
static_assert(sizeof(RayF) == sizeof(float) * 6, "Ray<float> size is not 24 bytes");

#include "Ray.hpp"

static constexpr RayF InvalidRayF = RayF(Zero3f, Zero3f);
static constexpr RayD InvalidRayD = RayD(Zero3d, Zero3d);

// // Ray Extern
// extern template class Ray<float>;
// extern template class Ray<double>;