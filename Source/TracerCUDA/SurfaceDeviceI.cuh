#pragma once

#include <cuda_runtime.h>
#include "RayLib/Vector.h"

enum class SurfaceType
{
	VOLUME,
	MESH
};


//struct SurfaceDeviceI
//{
//	//
//	SurfaceType						surfaceType;
//	void*							dataLocation;
//	
//	// Data Access Related
//	__device__ bool					HasData(const Vector3f& index) const;
//	__device__ Vector3				Velocity(const Vector3f& location) const;
//	__device__ float				Density(const Vector3f& location) const;
//
//	__device__ Vector3				Normal(const Vector3f& location) const;
//	__device__ Vector2				UV(const Vector3f& location) const;
//
//	// Device Special Function List
//	__device__ bool					HasDataVolume(const Vector3f& index) const;
//	__device__ Vector3				VelocityVolume(const Vector3f& location) const;
//	__device__ float				DensityVolume(const Vector3f& location) const;
//	__device__ Vector3				NormalVolume(const Vector3f& location) const;
//
//		
//	private:
//		typedef bool(SurfaceDeviceI::* HasDataFPtr)(const Vector3f&) const;		
//		typedef Vector3(SurfaceDeviceI::* VelocityFPtr)(const Vector3f&) const;
//		typedef float(SurfaceDeviceI::* DensityFPtr)(const Vector3f&) const;
//		typedef Vector3(SurfaceDeviceI::* NormalFPtr)(const Vector3f&) const;
//		typedef Vector3(SurfaceDeviceI::* UVFPtr)(const Vector3f&) const;
//
//
//		static constexpr HasDataFPtr HasDataFuncs[] = {&SurfaceDeviceI::HasDataVolume};		
//		static constexpr VelocityFPtr VolumeFuncs[] = {&SurfaceDeviceI::VelocityVolume};
//		static constexpr DensityFPtr DensityFuncs[] = {&SurfaceDeviceI::DensityVolume};
//		static constexpr NormalFPtr NormalFuncs[] = {&SurfaceDeviceI::NormalVolume};
//		static constexpr UVFPtr UVFuncs[] = {nullptr};
//};
//
////__device__
////inline bool SurfaceDeviceI::HasData(const Vector3f& index) const
////{
////	return HasDataFuncs[static_cast<int>(surfaceType)];
////}
////
//////__device__
//////inline Vector3 SurfaceDeviceI::Velocity(const Vector3f& index) const
//////{
//////
//////}
//////
//////__device__
//////inline float SurfaceDeviceI::Density(const Vector3f& index) const
//////{
//////
//////}
//////
//////__device__
//////inline Vector3  SurfaceDeviceI::Normal(const Vector3f& index) const
//////{
//////
//////}
////
////__device__ bool SurfaceDeviceI::HasDataVolume(const Vector3f& index) const
////{
////
////}