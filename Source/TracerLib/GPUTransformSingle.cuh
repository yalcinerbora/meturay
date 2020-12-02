#pragma once

#include "GPUTransformI.h"
#include "DeviceMemory.h"

class GPUTransformSingle : public GPUTransformI
{
    private:
		const Matrix4x4&		transform;
		const Matrix4x4&		invTransform;

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
		Vector3			LocalToWorld(const Vector3&, bool isDirection = false,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;
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
{}

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
inline Vector3 GPUTransformSingle::LocalToWorld(const Vector3& vec, bool isDirection,
												const uint32_t*, const float*, uint32_t) const
{
	Vector4 vector = Vector4(vec, (isDirection) ? 0.0f : 1.0f);
	return transform * vector;
}

__device__
inline QuatF GPUTransformSingle::ToLocalRotation(const uint32_t*, const float*,
												 uint32_t) const
{
	// TODO: fetch rotation portion of the matrix and convert it to quaternion
	return IdentityQuatF;
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
