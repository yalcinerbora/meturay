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
		virtual			~GPUTransformSingle() = default;

		__device__  __host__
		RayF			WorldToLocal(const RayF&,
									 const uint32_t* indices = nullptr, 
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;	
		__device__  __host__
		Vector3			LocalToWorld(const Vector3&,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const override;
		__device__  __host__
		QuatF			ToLocalRotation(const uint32_t* indices = nullptr,
										const float* weights = nullptr,
										uint32_t count = 0) const override;
};


class CPUTransformSingle : public CPUTransformI
{
	public:	
		static const char*			TypeName() { return "SingleTransform"; }
    private:
		DeviceMemory				memory;		
		const Matrix4x4*			dMatrices;
		const GPUTransformSingle*	dGPUTransforms;
		const GPUTransformList		transformList;

    protected:
    public:
		// Interface
		const char*					Type() const override;
		const GPUTransformList&		GPUTransforms() const override;
		SceneError					InitializeGroup(const NodeListing& transformNodes,
													double time,
													const std::string& scenePath) override;
		SceneError					ChangeTime(const NodeListing& transformNodes, double time,
											   const std::string& scenePath) override;

		size_t						UsedGPUMemory() const override;
		size_t						UsedCPUMemory() const override;
};

__device__  __host__
inline RayF GPUTransformSingle::WorldToLocal(const RayF& r,
											 const uint32_t* indices, const float* weights,
											 uint32_t count) const
{

}

 __device__  __host__
inline Vector3 GPUTransformSingle::LocalToWorld(const Vector3& vector,
												const uint32_t* indices, const float* weights,
												uint32_t count) const
{

}

__device__  __host__
inline QuatF GPUTransformSingle::ToLocalRotation(const uint32_t* indices, const float* weights,
												 uint32_t count) const
{

}

inline const char* CPUTransformSingle::Type() const
{
	return TypeName();
}

inline const GPUTransformList& CPUTransformSingle::GPUTransforms() const
{
	return transformList;
}

inline size_t CPUTransformSingle::UsedGPUMemory() const
{
	return memory.Size();
}

inline size_t CPUTransformSingle::UsedCPUMemory() const
{
	return 0;
}
