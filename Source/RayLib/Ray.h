#pragma once
/**

Ray struct for convenient usability.

*/
#include "Matrix.h"
#include "Vector.h"
#include "Quaternion.h"
#include "HybridFunctions.h"
template<class T, typename = ArithmeticEnable<T>>
class Ray;
template<class T>
class Ray<T>
{
    private:
        Vector<3,T>                                 direction;
        Vector<3,T>                                 position;

    protected:
    public:
        // Constructors & Destructor
        constexpr                                   Ray() = default;
        constexpr __device__ __host__               Ray(const Vector<3,T>& direction,
                                                        const Vector<3, T>& position);
        constexpr __device__ __host__               Ray(const Vector<3, T>[2]);
                                                    Ray(const Ray&) = default;
                                                    ~Ray() = default;
        Ray&                                        operator=(const Ray&) = default;

        // Assignment Operators
        __device__ __host__ Ray&                    operator=(const Vector<3, T>[2]);

        __device__ __host__ const Vector<3,T>&      getDirection() const;
        __device__ __host__ const Vector<3,T>&      getPosition() const;

         // Intersections
        __device__ __host__ bool                    IntersectsSphere(Vector<3, T>& pos, T& t,
                                                                     const Vector<3, T>& sphereCenter,
                                                                     T sphereRadius) const;
        __device__ __host__ bool                    IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                                                       const Vector<3, T> triCorners[3],
                                                                       bool cullFace = true) const;
        __device__ __host__ bool                    IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                                                       const Vector<3, T>& t0,
                                                                       const Vector<3, T>& t1,
                                                                       const Vector<3, T>& t2,
                                                                       bool cullFace = true) const;
        __device__ __host__ bool                    IntersectsPlane(Vector<3, T>& position, T& t,
                                                                    const Vector<3, T>& planePos,
                                                                    const Vector<3, T>& normal);
        __device__ __host__ bool                    IntersectsAABB(const Vector<3, T>& min,
                                                               const Vector<3, T>& max,
                                                               const Vector<2, T>& tMinMax = Vector<2, T>(-INFINITY, INFINITY)) const;
        __device__ __host__ bool                    IntersectsAABB(Vector<3,T>& pos, T& t,
                                                               const Vector<3, T>& min,
                                                               const Vector<3, T>& max,
                                                               const Vector<2, T>& tMinMax = Vector<2, T>(-INFINITY, INFINITY)) const;

        // Utility
        __device__ __host__ Ray                     Reflect(const Vector<3, T>& normal) const;
        __device__ __host__ Ray&                    ReflectSelf(const Vector<3, T>& normal);
        __device__ __host__ bool                    Refract(Ray& out, const Vector<3, T>& normal,
                                                            T fromMedium, T toMedium) const;
        __device__ __host__ bool                    RefractSelf(const Vector<3, T>& normal,
                                                                T fromMedium, T toMedium);

        // Randomization (Hemispherical)
        __device__ __host__ static Ray              RandomRayCosine(T xi0, T xi1,
                                                                    const Vector<3, T>& normal,
                                                                    const Vector<3, T>& position);
        __device__ __host__ static Ray              RandomRayUnfirom(T xi0, T xi1,
                                                                     const Vector<3, T>& normal,
                                                                     const Vector<3, T>& position);

        __device__ __host__ Ray                     NormalizeDir() const;
        __device__ __host__ Ray&                    NormalizeDirSelf();
        __device__ __host__ Ray                     Advance(T) const;
        __device__ __host__ Ray                     Advance(T t, const Vector<3,T>&) const;
        __device__ __host__ Ray&                    AdvanceSelf(T);
        __device__ __host__ Ray&                    AdvanceSelf(T t, const Vector<3, T>&);
        __device__ __host__ Ray                     Transform(const Quaternion<T>&) const;
        __device__ __host__ Ray                     Transform(const Matrix<3, T>&) const;
        __device__ __host__ Ray                     Transform(const Matrix<4, T>&) const;
        __device__ __host__ Ray                     TransformSelf(const Quaternion<T>&);
        __device__ __host__ Ray&                    TransformSelf(const Matrix<3, T>&);
        __device__ __host__ Ray&                    TransformSelf(const Matrix<4, T>&);
        __device__ __host__ Vector<3,T>             AdvancedPos(T t) const;
};

using RayF = Ray<float>;
using RayD = Ray<double>;

// Requirements of IERay
//static_assert(std::is_literal_type<RayF>::value == true, "Ray has to be literal type");
static_assert(std::is_trivially_copyable<RayF>::value == true, "Ray has to be trivially copyable");
static_assert(std::is_polymorphic<RayF>::value == false, "Ray should not be polymorphic");
static_assert(sizeof(RayF) == sizeof(float) * 6, "Ray<float> size is not 24 bytes");

#include "Ray.hpp"

static constexpr RayF InvalidRayF = RayF(Zero3f, Zero3f);
static constexpr RayD InvalidRayD = RayD(Zero3d, Zero3d);

// // Ray Extern
// extern template class Ray<float>;
// extern template class Ray<double>;