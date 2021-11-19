#pragma once

#include "RayLib/Ray.h"
#include "RayLib/AABB.h"

// This class is syntactically same with the GPUTransformIdentity
// However it does not linked to the inheritance chain of GPUTransformI
// OptiX hates virtual functions so we call our intersection routine
// with this

class GPUTransformEmpty
{
    private:
    protected:
    public:
		// Constructors & Destructor
						GPUTransformEmpty() = default;
						~GPUTransformEmpty() = default;

		__device__ __forceinline__
		RayF				WorldToLocal(const RayF&,
										 const uint32_t* indices = nullptr,
										 const float* weights = nullptr,
										 uint32_t count = 0) const;
		__device__ __forceinline__
		Vector3			WorldToLocal(const Vector3&, bool isDirection = false,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const;
		__device__ __forceinline__
		AABB3f			WorldToLocal(const AABB3f&) const;

		__device__ __forceinline__
		Vector3			LocalToWorld(const Vector3&, bool isDirection = false,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const;
		__device__ __forceinline__
		AABB3f			LocalToWorld(const AABB3f&) const;

		__device__ __forceinline__
		QuatF			ToLocalRotation(const uint32_t* indices = nullptr,
										const float* weights = nullptr,
										uint32_t count = 0) const;

		__device__ __forceinline__
		Matrix4x4		GetLocalToWorldAsMatrix() const;
};

__device__ __forceinline__
inline RayF GPUTransformEmpty::WorldToLocal(const RayF& r,
											const uint32_t*, const float*,
											uint32_t) const
{
	return r;
}

__device__ __forceinline__
inline Vector3f GPUTransformEmpty::WorldToLocal(const Vector3f& vec, bool,
												const uint32_t*, const float*,
												uint32_t) const
{
	return vec;
}

__device__ __forceinline__
inline AABB3f GPUTransformEmpty::WorldToLocal(const AABB3f& aabb) const
{
	return aabb;
}

__device__ __forceinline__
inline Vector3 GPUTransformEmpty::LocalToWorld(const Vector3& vector, bool,
											   const uint32_t*, const float*,
											   uint32_t) const
{
	return vector;
}

__device__ __forceinline__
inline AABB3f GPUTransformEmpty::LocalToWorld(const AABB3f& aabb) const
{
	return aabb;
}

__device__ __forceinline__
inline QuatF GPUTransformEmpty::ToLocalRotation(const uint32_t*, const float*,
												uint32_t) const
{
	return IdentityQuatF;
}

__device__ __forceinline__
inline Matrix4x4 GPUTransformEmpty::GetLocalToWorldAsMatrix() const
{
	return Indentity4x4;
}