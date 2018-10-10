#pragma once

/**

Lightweight texture wrapper for cuda

Object oriented design and openGL like access

*/

#include "CudaCheck.h"
#include "DeviceMemory.h"
#include "Vector.h"
#include "Types.h"
#include <cuda_runtime.h>

#include <cstddef>

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

static constexpr cudaTextureAddressMode DetermineAddressMode(EdgeResolveType);
static constexpr cudaTextureFilterMode DetermineFilterMode(InterpolationType);

template<int D, class T>
class Texture : public DeviceLocalMemoryI
{
	static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");
	static_assert(std::is_same<T, float>::value ||
				  std::is_same<T, float1>::value ||
				  std::is_same<T, float2>::value ||
				  std::is_same<T, float4>::value ||

				  std::is_same<T, char>::value ||
				  std::is_same<T, char1>::value ||
				  std::is_same<T, char2>::value ||
				  std::is_same<T, char4>::value ||

				  std::is_same<T, short>::value ||
				  std::is_same<T, short1>::value ||
				  std::is_same<T, short2>::value ||
				  std::is_same<T, short3>::value ||

				  std::is_same<T, int>::value ||
				  std::is_same<T, int1>::value ||
				  std::is_same<T, int2>::value ||
				  std::is_same<T, int3>::value ||

				  std::is_same<T, unsigned char>::value ||
				  std::is_same<T, uchar1>::value ||
				  std::is_same<T, uchar2>::value ||
				  std::is_same<T, uchar4>::value ||

				  std::is_same<T, unsigned short>::value ||
				  std::is_same<T, ushort1>::value ||
				  std::is_same<T, ushort2>::value ||
				  std::is_same<T, ushort3>::value ||

				  std::is_same<T, unsigned int>::value ||
				  std::is_same<T, uint1>::value ||
				  std::is_same<T, uint2>::value ||
				  std::is_same<T, uint3>::value, "Types should be those specified.");

	private:
		cudaMipmappedArray_t			data;
		cudaTextureObject_t				t;
		
	protected:
	public:
		__host__						Texture(InterpolationType,
												EdgeResolveType,
												bool unormType,
												const Vector<D, unsigned int>& dim);
		//__device__ __host__				Texture(InterpolationType,
		//										EdgeResolveType,
		//										bool unormType,
		//										const Vector<D + 1, unsigned int>& dim);

		// Accessors (D + 1 variants are uses mip map)
		__device__ const T				operator()(const Vector<D, float>&) const;
		__device__ const T				operator()(const Vector<D + 1, float>&) const;

		// Copy Data
		void							Copy(const Byte* sourceData,
											 const Vector<D, unsigned int>& size,
											 int mipLevel = 0);

		// Memory Migration
		void							MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) override;
};

//template<int D, class T>
//class TextureArray : public DeviceLocalMemoryI
//{
//	static_assert(D >= 1 && D <= 2, "At most 2D texture arrays are supported");
//
//	private:
//		cudaArray_t						data;
//		cudaTextureObject_t				t;
//
//	protected:
//		
//
//	public:
//		__host__				TextureArray(InterpolationType,
//													 EdgeResolveType,
//													 const Vector<D + 1, unsigned int>& dim);
//
//		__device__ 						operator cudaTextureObject_t();
//
//		// Accessors (D + 2 variants are uses mipmap)
//		__device__ const T				operator()(const Vector<D + 1, float>&) const;
//		__device__ const T				operator()(const Vector<D + 2, float>&) const;
//
//		// Special Access
//
//		// Memory Migration
//		void							MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) override;
//};
//
//template<class T>
//class TextureCube : public DeviceLocalMemoryI
//{
//	private:
//		cudaArray_t						data;
//		cudaTextureObject_t				t;
//		
//	protected:
//	public:
//		__host__				TextureCube(InterpolationType,
//													EdgeResolveType,
//													const Vector2ui& dim,
//													const T* data = nullptr);
//		__host__				TextureCube(InterpolationType,
//													EdgeResolveType,
//													const Vector3ui& dim,
//													const T* data = nullptr);
//		// Accessors (3D vector variant uses mipmap)
//		__device__ const T				operator()(const Vector2&) const;
//		__device__ const T				operator()(const Vector3&) const;
//
//		// Special Access
//
//		// Memory Migration
//		void							MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) override;
//};
//
//template< class T>
//class TextureCubeArray : public DeviceLocalMemoryI
//{
//	private:
//		cudaArray_t						data;
//		cudaTextureObject_t				t;
//		
//	protected:
//	public:
//		__host__				TextureCubeArray(InterpolationType,
//														 EdgeResolveType,
//														 const Vector3ui& dim,
//														 const T* data = nullptr);
//		__host__				TextureCubeArray(InterpolationType,
//														 EdgeResolveType,
//														 const Vector4ui& dim,
//														 const T* data = nullptr);
//		// Accessors (4D variant uses mipmap)
//		__device__ const T				operator()(const Vector3&) const;
//		__device__ const T				operator()(const Vector4&) const;
//
//		// Special Access
//
//
//		// Memory Migration
//		void							MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) override;
//};

// Ease of use Template Types
template<class T> using Texture1 = Texture<1, T>;
template<class T> using Texture2 = Texture<2, T>;
template<class T> using Texture3 = Texture<3, T>;

//template<class T> using Texture1Array = TextureArray<1, T>;
//template<class T> using Texture2Array = TextureArray<2, T>;

#include "Texture.hpp"