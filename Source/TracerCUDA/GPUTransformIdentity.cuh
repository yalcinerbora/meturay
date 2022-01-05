#pragma once

#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "AuxiliaryDataKernels.cuh"
#include "RayLib/SceneNodeNames.h"
#include "TypeTraits.h"

class GPUTransformIdentity final : public GPUTransformI
{
    private:
    protected:
    public:
		// Constructors & Destructor
						GPUTransformIdentity() = default;
		virtual			~GPUTransformIdentity() = default;

		__device__ __forceinline__
		RayF				WorldToLocal(const RayF&,
										 const uint32_t* indices = nullptr,
										 const float* weights = nullptr,
										 uint32_t count = 0) const override;
		__device__ __forceinline__
		Vector3			WorldToLocal(const Vector3&, bool isDirection = false,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;
		__device__ __forceinline__
		AABB3f			WorldToLocal(const AABB3f&) const override;

		__device__ __forceinline__
		Vector3			LocalToWorld(const Vector3&, bool isDirection = false,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;
		__device__ __forceinline__
		AABB3f			LocalToWorld(const AABB3f&) const override;

		__device__ __forceinline__
		QuatF			ToLocalRotation(const uint32_t* indices = nullptr,
										const float* weights = nullptr,
										uint32_t count = 0) const override;

		__device__ __forceinline__
		Vector3f			ToWorldScale(const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;
		__device__ __forceinline__
		Vector3f			ToLocalScale(const uint32_t* indices = nullptr,
								     const float* weights = nullptr,
								     uint32_t count = 0) const override;

		__device__ __forceinline__
		Matrix4x4		GetLocalToWorldAsMatrix() const override;
};

class CPUTransformIdentity : public CPUTransformGroupI
{
	public:
		static const char*			TypeName() { return NodeNames::TRANSFORM_IDENTITY; }
    private:
		DeviceMemory				memory;
		const GPUTransformIdentity*	dGPUTransforms;
		GPUTransformList			gpuTransformList;

    protected:
    public:
		// Constructors & Destructor
									CPUTransformIdentity() = default;
		virtual						~CPUTransformIdentity() = default;

		// Interface
		const char*					Type() const override;
		const GPUTransformList&		GPUTransforms() const override;
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

__device__ __forceinline__
RayF GPUTransformIdentity::WorldToLocal(const RayF& r,
										const uint32_t*, const float*,
										uint32_t) const
{
	return r;
}

__device__ __forceinline__
Vector3f GPUTransformIdentity::WorldToLocal(const Vector3f& vec, bool,
											const uint32_t*, const float*,
											uint32_t) const
{
	return vec;
}

__device__ __forceinline__
AABB3f GPUTransformIdentity::WorldToLocal(const AABB3f& aabb) const
{
	return aabb;
}

__device__ __forceinline__
Vector3 GPUTransformIdentity::LocalToWorld(const Vector3& vector, bool,
										   const uint32_t*, const float*,
										   uint32_t) const
{
	return vector;
}

__device__ __forceinline__
AABB3f GPUTransformIdentity::LocalToWorld(const AABB3f& aabb) const
{
	return aabb;
}

__device__ __forceinline__
QuatF GPUTransformIdentity::ToLocalRotation(const uint32_t*, const float*,
											uint32_t) const
{
	return IdentityQuatF;
}

__device__ __forceinline__
Vector3f GPUTransformIdentity::ToWorldScale(const uint32_t* indices,
											const float* weights, uint32_t count) const
{
	return Vector3f(1.0f);
}

__device__ __forceinline__
Vector3f GPUTransformIdentity::ToLocalScale(const uint32_t* indices,
										    const float* weights,
										    uint32_t count) const
{
	return Vector3f(1.0f);
}

__device__ __forceinline__
Matrix4x4 GPUTransformIdentity::GetLocalToWorldAsMatrix() const
{
	return Indentity4x4;
}

inline const char* CPUTransformIdentity::Type() const
{
	return TypeName();
}

inline SceneError CPUTransformIdentity::InitializeGroup(const NodeListing& transformNodes,
														double,
												 		const std::string&)
{
	if(transformNodes.size() != 1 &&
	   (*transformNodes.begin())->IdCount() != 1)
		return SceneError::TRANSFORM_TYPE_INTERNAL_ERROR;

	GPUMemFuncs::EnlargeBuffer(memory, sizeof(GPUTransformIdentity));
	dGPUTransforms = static_cast<GPUTransformIdentity*>(memory);
	return SceneError::OK;
}

inline SceneError CPUTransformIdentity::ChangeTime(const NodeListing&, double,
												   const std::string&)
{
	return SceneError::OK;
}

inline TracerError CPUTransformIdentity::ConstructTransforms(const CudaSystem& system)
{
	// Call allocation kernel
	const CudaGPU& gpu = system.BestGPU();
	CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
	gpu.GridStrideKC_X(0, 0, TransformCount(),
					   //
					   KCConstructGPUClass<GPUTransformIdentity>,
					   //
					   const_cast<GPUTransformIdentity*>(dGPUTransforms),
					   TransformCount());

	gpu.WaitMainStream();

	// Generate transform list
	for(uint32_t i = 0; i < TransformCount(); i++)
	{
		const auto* ptr = static_cast<const GPUTransformI*>(dGPUTransforms + i);
		gpuTransformList.push_back(ptr);
	}
	return TracerError::OK;
}

inline const GPUTransformList& CPUTransformIdentity::GPUTransforms() const
{
	return gpuTransformList;
}

inline uint32_t CPUTransformIdentity::TransformCount() const
{
	return 1;
}

inline size_t CPUTransformIdentity::UsedGPUMemory() const
{
	return memory.Size();
}

inline size_t CPUTransformIdentity::UsedCPUMemory() const
{
	return 0;
}

static_assert(IsTracerClass<CPUTransformIdentity>::value,
			  "CPUTransformIdentity is not a tracer class");