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
		__device__ __host__
						GPUTransformSingle(const Matrix4x4& transform,
										   const Matrix4x4& invTransform);
		virtual			~GPUTransformSingle() = default;

		__device__ __host__
		RayF			WorldToLocal(const RayF&,
									 const uint32_t* indices = nullptr, 
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;	
		__device__ __host__
		Vector3			LocalToWorld(const Vector3&,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;
		__device__ __host__
		QuatF			ToLocalRotation(const uint32_t* indices = nullptr,
										const float* weights = nullptr,
										uint32_t count = 0) const override;
};

using CPUTransformList = std::vector<GPUTransformSingle>;

class CPUTransformSingle : public CPUTransformGroupI
{
	public:	
		static const char*				TypeName() { return "SingleTransform"; }

		static constexpr const char*	LAYOUT_MATRIX	= "matrix4x4";
		static constexpr const char*	LAYOUT_TRS		= "trs";

		static constexpr const char*	LAYOUT			= "layout";
		static constexpr const char*	MATRIX			= "matrix";
		static constexpr const char*	TRANSLATE		= "translate";
		static constexpr const char*	ROTATE			= "rotate";
		static constexpr const char*	SCALE			= "scale";

    private:
		DeviceMemory				memory;		
		const Matrix4x4*			dTransforms;
		const Matrix4x4*			dInvTransforms;
		const GPUTransformSingle*	dGPUTransforms;
		
		CPUTransformList			hGPUTransforms;
		GPUTransformList			gpuTransformList;
		GPUTransformList			cpuTransformList;
		uint32_t					transformCount;

    protected:
    public:
		// Interface
		const char*					Type() const override;
		const GPUTransformList&		GPUTransforms() const override;
		const GPUTransformList&		CPUTransforms() const override;
		SceneError					InitializeGroup(const NodeListing& transformNodes,
													double time,
													const std::string& scenePath) override;
		SceneError					ChangeTime(const NodeListing& transformNodes, double time,
											   const std::string& scenePath) override;
		TracerError					ConstructTransforms(const CudaSystem&) override;
		uint32_t					TransformCount() const override;

		size_t						UsedGPUMemory() const override;
		size_t						UsedCPUMemory() const override;
};

__device__ __host__
GPUTransformSingle::GPUTransformSingle(const Matrix4x4& transform,
									   const Matrix4x4& invTransform)
	: transform(transform)
	, invTransform(invTransform)
{}

__device__ __host__
inline RayF GPUTransformSingle::WorldToLocal(const RayF& r,
											 const uint32_t* indices, const float* weights,
											 uint32_t count) const
{
	return r.Transform(invTransform);
}

__device__ __host__
inline Vector3 GPUTransformSingle::LocalToWorld(const Vector3& vector,
												const uint32_t* indices, const float* weights,
												uint32_t count) const
{
	 return transform * vector;
}

__device__ __host__
inline QuatF GPUTransformSingle::ToLocalRotation(const uint32_t* indices, const float* weights,
												 uint32_t count) const
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

inline const GPUTransformList& CPUTransformSingle::CPUTransforms() const
{
	return cpuTransformList;
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
