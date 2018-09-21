#pragma once

#include "RayLib/VolumeI.h"
#include "RayLib/DeviceMemory.h"
#include "RayLib/Vector.h"
#include "RayLib/Texture.cuh"

#include "RayLib/MayaCacheIO.h"

#include "SVODevice.cuh"

//struct VolumeDeviceData
//{
//	//Vector4*						data;
//	Vector3ui						size;
//	Vector3f						worldSize;
//	
//	Texture3<float4>				volumeTex;
//
//	uint32_t						surfaceId;
//	uint32_t						materialId;
//
//	// Size Related
//	__device__ uint64_t				LinearIndex(const Vector3ui& index) const;
//	__device__ uint64_t				LinearSize() const;
//	__device__ Vector3ui			Size() const;
//	// Data Access Related
//	__device__ bool					HasData(const Vector3ui& index) const;
//	__device__ Vector3				Velocity(const Vector3ui& index) const;
//	__device__ float				Density(const Vector3ui& index) const;
//};

//class VolumeGPU : public VolumeI, public VolumeDeviceData
//{
//	private:			
//	protected:
//		std::string						fileName;
//	
//		//DeviceMemory					mem;		
//		SVODevice						svo;
//
//	public:
//		// Constructors & Destructor
//										VolumeGPU(const std::string& fileName, 
//												  uint32_t materialId, 
//												  uint32_t surfaceId);
//										VolumeGPU(const VolumeGPU&) = delete;
//										VolumeGPU(VolumeGPU&&) = default;
//		VolumeGPU&						operator=(const VolumeGPU&) = delete;
//		VolumeGPU&						operator=(VolumeGPU&&) = default;
//										~VolumeGPU() = default;
//										
//		//
//		Vector3ui						Size() const override;
//		const std::string&				FileName() const override;
//
//		// From Surface Interface
//		uint32_t						Id() const override;
//		uint32_t						MaterialId() const override;
//		void							HitRays(const HitRecordGMem,
//												const ConstRayRecordGMem,
//												uint64_t rayCount) const override;
//
//
//		// Delete
//		const SVODevice&				SVO() const { return svo; }
//		
//		
//};
//
//// Different classes for different implementations of data types
//class NCVolumeGPU final : public VolumeGPU
//{
//	private:
//		MayaCache::MayaNSCacheInfo		info;
//
//	public:
//		// Constructors & Destructor
//										NCVolumeGPU(const std::string& fileName, 
//												  uint32_t materialId, 
//												  uint32_t surfaceId);
//										NCVolumeGPU(const NCVolumeGPU&) = delete;
//										NCVolumeGPU(NCVolumeGPU&&) = default;
//		NCVolumeGPU&					operator=(const NCVolumeGPU&) = delete;
//		NCVolumeGPU&					operator=(NCVolumeGPU&&) = default;
//										~NCVolumeGPU() = default;
//
//		// From Volume Interface
//		Error							Load() override;
//
//		// From Animate Interface
//		Error							ChangeFrame(double time) override;
//
//};
//
//__device__ 
//inline uint64_t VolumeDeviceData::LinearIndex(const Vector3ui& index) const
//{
//	return	index[2] * size[1] * size[0] +
//			index[1] * size[0] +
//			index[0];
//}
//
//__device__ 
//inline uint64_t VolumeDeviceData::LinearSize() const
//{
//	return size[0] * size[1] * size[2];
//}
//
//__device__
//inline Vector3ui VolumeDeviceData::Size() const
//{
//	return size;
//}
//
//__device__ 
//inline bool VolumeDeviceData::HasData(const Vector3ui& index) const
//{
//	return (Density(index) != 0.0f);
//}
//
//__device__
//inline Vector3 VolumeDeviceData::Velocity(const Vector3ui& index) const
//{
//	return Zero3;// Vector3f(reinterpret_cast<float*>(data) + LinearIndex(index));
//}
//
//__device__ 
//inline float VolumeDeviceData::Density(const Vector3ui& index) const
//{
//	//float d = (*(data + LinearIndex(index)))[3];
//	return 0.0f; //(d != 0.0f);
//}