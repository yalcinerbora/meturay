#pragma once

#include "Matrix.h"
#include "Vector.h"

class Ray
{
	private:
		Vector3								direction;
		Vector3								position;

	protected:
	public:
		// Constructors & Destructor
		constexpr __device__ __host__		Ray();
		constexpr __device__ __host__		Ray(float dX, float dY, float dZ,
												float pX, float pY, float pZ);
		__device__ __host__					Ray(const Vector3& direction, const Vector3& position);
		__device__ __host__					Ray(const Vector3[2]);
											Ray(const Ray&) = default;
											~Ray() = default;
		Ray&								operator=(const Ray&) = default;

		// Assignment Operators
		__device__ __host__ Ray&			operator=(const Vector3[2]);

		__device__ __host__ const Vector3&	getDirection() const;
		__device__ __host__ const Vector3&	getPosition() const;

		// Intersections
		__device__ __host__ bool			IntersectsSphere(Vector3& pos,
															 float& t,
															 const Vector3& sphereCenter,
															 float sphereRadius) const;
		__device__ __host__ bool			IntersectsTriangle(Vector3& baryCoords,
															   float& t,
															   bool cullFace,
															   const Vector3 triCorners[3]) const;
		__device__ __host__ bool			IntersectsTriangle(Vector3& baryCoords,
															   float& t,
															   bool cullFace,
															   const Vector3& t0,
															   const Vector3& t1,
															   const Vector3& t2) const;
		__device__ __host__ bool			IntersectsAABB(const Vector3& min,
														   const Vector3& max) const;

		// Utility
		__device__ __host__ Ray				Reflect(const Vector3& normal) const;
		__device__ __host__ Ray&			ReflectSelf(const Vector3& normal);
		__device__ __host__ bool			Refract(Ray& out, const Vector3& normal,
													float fromMedium, float toMedium) const;
		__device__ __host__ bool			RefractSelf(const Vector3& normal,
														float fromMedium, float toMedium);
		
		// Randomization (Hemi spherical)
		__device__ __host__ static Ray		RandomRayCosine(float xi0, float xi1,
															const Vector3& normal,
															const Vector3& position);
		__device__ __host__ static Ray		RandomRayUnfirom(float xi0, float xi1,
															 const Vector3& normal,
															 const Vector3& position);

		__device__ __host__ Ray				NormalizeDir() const;
		__device__ __host__ Ray&			NormalizeDirSelf();
		__device__ __host__ Ray				Advance(float) const;
		__device__ __host__ Ray&			AdvanceSelf(float);
		__device__ __host__ Ray				Transform(const Matrix4x4&) const;
		__device__ __host__ Ray&			TransformSelf(const Matrix4x4&);
		__device__ __host__ Vector3			AdvancedPos(float t) const;
};

// Requirements of IERay
static_assert(std::is_literal_type<Ray>::value == true, "IERay has to be literal type");
static_assert(std::is_trivially_copyable<Ray>::value == true, "IERay has to be trivially copyable");
static_assert(std::is_polymorphic<Ray>::value == false, "IERay should not be polymorphic");
static_assert(sizeof(Ray) == sizeof(float) * 6, "IERay size is not 24 bytes");

#include "Ray.hpp"