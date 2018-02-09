#pragma once

/**

Interpolatable Multi-dimensional (1D, 2D, 3D) dense data

Specifically primitive strucutre (no templates on class, etc.)
Textures are not responsible for data allocation, only wraps access
according to the template parameters.

All textures support potentially mipped dataset

*/

#include "CudaCheck.h"
#include "Vector.h"

enum class EdgeResolveType
{
	CLAMP,
	WRAP,
	MIRROR
};

enum class InterpolationType
{
	NEAREST,
	LINEAR,
	CUBIC
};

enum class TextureLayoutType
{
	ROW_MAJOR,
	COLUMN_MAJOR,
	Z_ORDER
};

struct TextureData
{
	Vector3i		dim;
	int				mipCount; // Base texture is also a mip (this should at least 1)
	const void*		data;
};

struct Sampler
{
	EdgeResolveType		E;
	InterpolationType	I;
	TextureLayoutType	T;
	//	
	Vector3i			offset;
	Vector3i			scale;
	// Access (scale[i] * i + offset[i]) for a dimension
};

template<int D>
class Texture
{
	static_assert(D >= 1 && D <= 3, "At most 3D arrays are supported");

	private:
		const TextureData&						t;
		const Sampler&							s;

		__device__ __host__ int64_t				DimToLinear(int i, int j, int k);

	protected:
	public:
		__device__ __host__						Texture(const TextureData&,
														const Sampler&);
		// Accessors (D + 1 variants are uses)
		// Direct Index Based Access
		template<class T>
		__device__ __host__ const T				operator()(const Vector<D, int>&) const;
		//template<class T>
		__device__ __host__ const float			operator()(const Vector<D + 1, int>&) const;
		// Normalized [0-1] Access
		//template<class T>
		__device__ __host__ const float			operator()(const Vector<D, float>&) const;
		//template<class T>
		__device__ __host__ const float			operator()(const Vector<D + 1, float>&) const;
};

// 
using Texture1 = Texture<1>;
using Texture2 = Texture<2>;
using Texture3 = Texture<3>;
using Texture4 = Texture<4>;

#include "Texture.hpp"