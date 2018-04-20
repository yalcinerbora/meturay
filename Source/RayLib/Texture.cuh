#pragma once

/**

Lightweight texture wrapper for cuda

Object oriented design and openGL like access

*/

#include "CudaCheck.h"
#include "DeviceMemory.h"
#include <cuda.h>
//#include <texture_object.h>
#include "Vector.h"

enum class InterpolationType
{
	NEAREST,
	LINEAR	
};

enum class EdgeResolveType
{
	WRAP,
	CLAMP,
	MIRROR
	// Border does not work properly
};

template<int D, class T>
class Texture : public DeviceLocalMemoryI
{
	static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");

	private:
		cudaMipmappedArray_t			data;
		cudaTextureObject_t				t;
		
	protected:
	public:
		__device__ __host__				Texture(InterpolationType,
												EdgeResolveType,
												const Vector<D, int>& dim,
												const T* data = nullptr);
		__device__ __host__				Texture(InterpolationType,
												EdgeResolveType,
												const Vector<D + 1, int>& dim,
												const T* data = nullptr);

		// Accessors (D + 1 variants are uses mip map)
		__device__ const T				operator()(const Vector<D, float>&) const;
		__device__ const T				operator()(const Vector<D + 1, float>&) const;

		// Special Access

		// Memory Migration
		void							MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) override;
};

template<int D, class T>
class TextureArray : public DeviceLocalMemoryI
{
	static_assert(D >= 1 && D <= 2, "At most 2D texture arrays are supported");

	private:
		cudaMipmappedArray_t			data;
		cudaTextureObject_t				t;

	protected:
		

	public:
		__device__ __host__				TextureArray(InterpolationType,
													 EdgeResolveType,
													 const Vector<D + 1, int>& dim,
													 const T* data = nullptr);
		__device__ __host__				TextureArray(InterpolationType,
													 EdgeResolveType,
													 const Vector<D + 2, int>& dim,
													 const T* data = nullptr);

		__device__ 						operator cudaTextureObject_t();

		// Accessors (D + 2 variants are uses mipmap)
		__device__ const T				operator()(const Vector<D + 1, float>&) const;
		__device__ const T				operator()(const Vector<D + 2, float>&) const;

		// Special Access

		// Memory Migration
		void							MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) override;
};

template<class T>
class TextureCube : public DeviceLocalMemoryI
{
	private:
		cudaMipmappedArray_t			data;
		cudaTextureObject_t				t;
		
	protected:
	public:
		__device__ __host__				TextureCube(InterpolationType,
													EdgeResolveType,
													const Vector2i& dim,
													const T* data = nullptr);
		__device__ __host__				TextureCube(InterpolationType,
													EdgeResolveType,
													const Vector3i& dim,
													const T* data = nullptr);
		// Accessors (3D vector variant uses mipmap)
		__device__ const T				operator()(const Vector2&) const;
		__device__ const T				operator()(const Vector3&) const;

		// Special Access

		// Memory Migration
		void							MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) override;
};

template< class T>
class TextureCubeArray : public DeviceLocalMemoryI
{
	private:
		cudaMipmappedArray_t			data;
		cudaTextureObject_t				t;
		
	protected:
	public:
		__device__ __host__				TextureCubeArray(InterpolationType,
														 EdgeResolveType,
														 const Vector3i& dim,
														 const T* data = nullptr);
		__device__ __host__				TextureCubeArray(InterpolationType,
														 EdgeResolveType,
														 const Vector4i& dim,
														 const T* data = nullptr);
		// Accessors (4D variant uses mipmap)
		__device__ const T				operator()(const Vector3&) const;
		__device__ const T				operator()(const Vector4&) const;

		// Special Access


		// Memory Migration
		void							MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) override;
};

// Ease of use Template Types
template<class T> using Texture1 = Texture<1, T>;
template<class T> using Texture2 = Texture<2, T>;
template<class T> using Texture3 = Texture<3, T>;

template<class T> using Texture1Array = TextureArray<1, T>;
template<class T> using Texture2Array = TextureArray<2, T>;

#include "Texture.hpp"