#pragma once

#include "Matrix.h"
#include "Vector.h"
#include "RayHitStructs.h"

template<class T, typename = ArithmeticEnable<T>>
class Ray;

template<class T>
class Ray<T>
{
	private:
		Vector<3,T>									direction;
		Vector<3,T>									position;

	protected:
	public:
		// Constructors & Destructor
		constexpr __device__ __host__				Ray(const Vector<3,T>& direction, const Vector<3, T>& position);
		constexpr __device__ __host__				Ray(const Vector3[2]);
													Ray(const Ray&) = default;
													~Ray() = default;
		Ray&										operator=(const Ray&) = default;

		// Assignment Operators
		__device__ __host__ Ray&					operator=(const Vector3[2]);

		__device__ __host__ const Vector<3,T>&		getDirection() const;
		__device__ __host__ const Vector<3,T>&		getPosition() const;

		// Intersections
		__device__ __host__ bool					IntersectsSphere(Vector<3, T>& pos, float& t,
																	 const Vector<3, T>& sphereCenter,
																	 float sphereRadius) const;
		__device__ __host__ bool					IntersectsTriangle(Vector<3, T>& baryCoords, float& t,
																	   const Vector<3, T> triCorners[3],
																	   bool cullFace = true) const;
		__device__ __host__ bool					IntersectsTriangle(Vector<3, T>& baryCoords, float& t,																	   
																	   const Vector<3, T>& t0,
																	   const Vector<3, T>& t1,
																	   const Vector<3, T>& t2,
																	   bool cullFace = true) const;
		__device__ __host__ bool					IntersectsAABB(const Vector<3, T>& min,
																   const Vector<3, T>& max) const;
		__device__ __host__ bool					IntersectsAABB(Vector<3,T>& pos, float& t,
																   const Vector<3, T>& min,
																   const Vector<3, T>& max) const;

		// Utility
		__device__ __host__ Ray						Reflect(const Vector<3, T>& normal) const;
		__device__ __host__ Ray&					ReflectSelf(const Vector<3, T>& normal);
		__device__ __host__ bool					Refract(Ray& out, const Vector<3, T>& normal,
															float fromMedium, float toMedium) const;
		__device__ __host__ bool					RefractSelf(const Vector<3, T>& normal,
																float fromMedium, float toMedium);
		
		// Randomization (Hemi spherical)
		__device__ __host__ static Ray				RandomRayCosine(float xi0, float xi1,
																	const Vector<3, T>& normal,
																	const Vector<3, T>& position);
		__device__ __host__ static Ray				RandomRayUnfirom(float xi0, float xi1,
																	 const Vector<3, T>& normal,
																	 const Vector<3, T>& position);

		__device__ __host__ Ray						NormalizeDir() const;
		__device__ __host__ Ray&					NormalizeDirSelf();
		__device__ __host__ Ray						Advance(float) const;
		__device__ __host__ Ray&					AdvanceSelf(float);
		__device__ __host__ Ray						Transform(const Matrix4x4&) const;
		__device__ __host__ Ray&					TransformSelf(const Matrix4x4&);
		__device__ __host__ Vector<3,T>				AdvancedPos(float t) const;
};

using RayF = Ray<float>;
using RayD = Ray<double>;

// Requirements of IERay
static_assert(std::is_literal_type<RayF>::value == true, "Ray has to be literal type");
static_assert(std::is_trivially_copyable<RayF>::value == true, "Ray has to be trivially copyable");
static_assert(std::is_polymorphic<RayF>::value == false, "Ray should not be polymorphic");
static_assert(sizeof(RayF) == sizeof(float) * 6, "Ray<float> size is not 24 bytes");

#include "Ray.hpp"