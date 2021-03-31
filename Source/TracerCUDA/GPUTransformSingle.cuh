#pragma once

#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"

class GPUTransformSingle : public GPUTransformI
{
    private:
		const Matrix4x4&		transform;
		const Matrix4x4&		invTransform;

		// Rotation only parts
		QuatF					rotation;
		QuatF					invRotation;

    protected:
    public:
		// Constructors & Destructor
		__device__
						GPUTransformSingle(const Matrix4x4& transform,
										   const Matrix4x4& invTransform);
		virtual			~GPUTransformSingle() = default;

		__device__
		RayF			WorldToLocal(const RayF&,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;
		__device__
		Vector3			WorldToLocal(const Vector3&, bool isDirection = false,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;
		__device__
		AABB3f			WorldToLocal(const AABB3f&) const override;

		__device__
		Vector3			LocalToWorld(const Vector3&, bool isDirection = false,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;
		__device__
		AABB3f			LocalToWorld(const AABB3f&) const override;

		__device__
		QuatF			ToLocalRotation(const uint32_t* indices = nullptr,
										const float* weights = nullptr,
										uint32_t count = 0) const override;
};

class CPUTransformSingle : public CPUTransformGroupI
{
	public:
		static const char*				TypeName() { return "Single"; }

		static constexpr const char*	LAYOUT_MATRIX	= "matrix4x4";
		static constexpr const char*	LAYOUT_TRS		= "trs";

		static constexpr const char*	LAYOUT			= "layout";
		static constexpr const char*	MATRIX			= "matrix";
		static constexpr const char*	TRANSLATE		= "translate";
		static constexpr const char*	ROTATE			= "rotate";
		static constexpr const char*	SCALE			= "scale";

    private:
		DeviceMemory					memory;
		const Matrix4x4*				dTransformMatrices;
		const Matrix4x4*				dInvTransformMatrices;
		const GPUTransformSingle*		dGPUTransforms;
		GPUTransformList				gpuTransformList;
		uint32_t                        transformCount;

    protected:
    public:
		// Constructors & Destructor
										CPUTransformSingle() = default;
		virtual							~CPUTransformSingle() = default;

		// Interface
		const char*						Type() const override;
		const GPUTransformList&			GPUTransforms() const override;
		SceneError						InitializeGroup(const NodeListing& transformNodes,
														double time,
														const std::string& scenePath) override;
		SceneError						ChangeTime(const NodeListing& transformNodes, double time,
												   const std::string& scenePath) override;
		TracerError						ConstructTransforms(const CudaSystem&) override;
		uint32_t						TransformCount() const override;

		size_t							UsedGPUMemory() const override;
		size_t							UsedCPUMemory() const override;
};

__device__
inline GPUTransformSingle::GPUTransformSingle(const Matrix4x4& transform,
											  const Matrix4x4& invTransform)
	: transform(transform)
	, invTransform(invTransform)
{
	// Top Left 3x3 Matrix defines rotation which is space defining 3 vectors
	Vector3f x(transform(0, 0), transform(0, 1), transform(0, 2));
	Vector3f y(transform(1, 0), transform(1, 1), transform(1, 2));
	Vector3f z(transform(2, 0), transform(2, 1), transform(2, 2));
	// Normalize space definiting linearly indepenent
	// and orthogonal vectors
	// These may not be unit vectors since transformation
	// may include scale factor
	x.NormalizeSelf();
	y.NormalizeSelf();
	z.NormalizeSelf();

	TransformGen::Space(invRotation, x, y, z);
	rotation = invRotation.Conjugate();
}

__device__
inline RayF GPUTransformSingle::WorldToLocal(const RayF& r,
											 const uint32_t*,
											 const float*,
											 uint32_t) const
{
	return r.Transform(invTransform);
}

__device__
inline Vector3 GPUTransformSingle::WorldToLocal(const Vector3& vec, bool isDirection,
												const uint32_t*, const float*,
												uint32_t) const
{
	Vector4 vector = Vector4(vec, (isDirection) ? 0.0f : 1.0f);
	return invTransform * vector;
}

__device__
inline AABB3f GPUTransformSingle::WorldToLocal(const AABB3f& aabb) const
{
	AABB3f result = NegativeAABB3f;
	for(int i = 0; i < AABB3f::AABBVertexCount; i++)
	{
		Vector4f vertex;
		vertex[0] = ((i >> 0 & 0x1) ? aabb.Max() : aabb.Min())[0];
		vertex[1] = ((i >> 1 & 0x1) ? aabb.Max() : aabb.Min())[1];
		vertex[2] = ((i >> 2 & 0x1) ? aabb.Max() : aabb.Min())[2];
		vertex[3] = 1.0f;

		vertex = invTransform * vertex;
		result.SetMax(Vector3f::Max(result.Max(), vertex));
		result.SetMin(Vector3f::Min(result.Min(), vertex));
	}
	return result;
}

__device__
inline Vector3 GPUTransformSingle::LocalToWorld(const Vector3& vec, bool isDirection,
												const uint32_t*, const float*, uint32_t) const
{
	Vector4 vector = Vector4(vec, (isDirection) ? 0.0f : 1.0f);
	return transform * vector;
}

__device__
inline AABB3f GPUTransformSingle::LocalToWorld(const AABB3f& aabb) const
{
	AABB3f result = NegativeAABB3f;
	for(int i = 0; i < AABB3f::AABBVertexCount; i++)
	{
		Vector4f vertex;
		vertex[0] = ((i >> 0 & 0x1) ? aabb.Max() : aabb.Min())[0];
		vertex[1] = ((i >> 1 & 0x1) ? aabb.Max() : aabb.Min())[1];
		vertex[2] = ((i >> 2 & 0x1) ? aabb.Max() : aabb.Min())[2];
		vertex[3] = 1.0f;

		vertex = transform * vertex;

		result.SetMax(Vector3f::Max(result.Max(), vertex));
		result.SetMin(Vector3f::Min(result.Min(), vertex));
	}
	return result;
}

__device__
inline QuatF GPUTransformSingle::ToLocalRotation(const uint32_t*, const float*,
												 uint32_t) const
{
	return invRotation;
}

inline const char* CPUTransformSingle::Type() const
{
	return TypeName();
}

inline const GPUTransformList& CPUTransformSingle::GPUTransforms() const
{
	return gpuTransformList;
}

inline uint32_t CPUTransformSingle::TransformCount() const
{
	return transformCount;
}

inline size_t CPUTransformSingle::UsedGPUMemory() const
{
	return memory.Size();
}

inline size_t CPUTransformSingle::UsedCPUMemory() const
{
	return 0;
}

static_assert(IsTracerClass<CPUTransformSingle>::value,
			  "CPUTransformSingle is not a tracer class");