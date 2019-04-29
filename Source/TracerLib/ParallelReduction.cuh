#pragma once
/**
Parallel Reduction Meta Implementation
with Templates

Can define custom reduce function and custom type

Modern Paralel Reduction Code
Utilizing new Kepler warp data transfer discussed here
http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
*/

#include "RayLib/CudaCheck.h"
#include "DeviceMemory.h"
#include "CudaConstants.h"
#include "ReduceFunctions.cuh"

template <class Type, ReduceFunc<Type> F>
__device__ inline void WarpReduce(Type& val)
{
	static_assert(sizeof(Type) % sizeof(int) == 0, "Type should fit into integers perfectly.");

	// Here using constexpr WarpSize to hint compiler to unroll this loop
	// "warpSize" implicit constant resides on const memory so compiler does not expand this loop
	#pragma unroll
	for(int offset = WarpSize / 2; offset > 0; offset >>= 1)
	{
		Type t;
		int* to = reinterpret_cast<int*>(&t);
		int* from = reinterpret_cast<int*>(&val);

		#pragma unroll
		for(int i = 0; i < static_cast<int>(sizeof(Type) / sizeof(int)); i++)
		{
			to[i] = __shfl_down_sync(0xFFFFFFFF, from[i], offset);
		}
		// Actual Warp
		val = F(val, t);
	}
}

// Block Reduce
template <class Type, ReduceFunc<Type> F>
__device__ inline void BlockReduce(Type& val, Type identityElement)
{
	extern __shared__ double2 smem[];
	Type* shared = reinterpret_cast<Type*>(smem);

	int lane = threadIdx.x % warpSize;	// Nvidia uses lane term for warp threads
	int wId = threadIdx.x / warpSize;	// Warp Id in the block

										// Switch Depending on the component
										// Each warp will reduce 32 to 1
	WarpReduce<Type, F>(val);

	// Leader of each warp (in this case lane 0) write to sMem
	if(lane == 0) shared[wId] = val;
	__syncthreads();

	// Read from reduced memory if that warp was active
	// or write identity reduction element
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : identityElement;

	// Last Reduction using first warp
	if(wId == 0) WarpReduce<Type, F>(val);
}

// Block Reduce for tex call (2d)
template <class Type, ReduceFunc<Type> F>
__device__ inline void BlockReduce2D(Type& val, Type identityElement)
{
	extern __shared__ double2 smem[];
	Type* shared = reinterpret_cast<Type*>(smem);

	int id1D = threadIdx.y * blockDim.x + threadIdx.x;
	int lane = id1D % warpSize;							// Nvidia uses lane term for warp threads
	int wId = id1D / warpSize;							// Warp Id in the block

														// Switch Depending on the component
														// Each warp will reduce 32 to 1
	WarpReduce<Type, F>(val);

	// Leader of each warp (in this case lane 0) write to sMem
	if(lane == 0) shared[wId] = val;
	__syncthreads();

	// Read from reduced memory if that warp was active
	// or write identity reduction element
	val = (id1D < (blockDim.x * blockDim.y) / warpSize) ? shared[lane] : identityElement;

	// Last Reduction using first warp
	if(wId == 0) WarpReduce<Type, F>(val);
}

template <class Type, ReduceFunc<Type> F>
__global__ void ParalelReduction(Type* gOut,
								 const Type* gIn,
								 unsigned int totalCount,
								 Type identityElement)
{
	Type result = identityElement;
	unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	if(globalId < totalCount) result = gIn[globalId];

	BlockReduce<Type, F>(result, identityElement);
	if(threadIdx.x == 0) gOut[blockIdx.x] = result;
}

template <class Type, ReduceFunc<Type> F>
__global__ void ParalelReductionTex(Type* gOut,
									cudaTextureObject_t tIn,
									uint2 dimensions, uint2 offset,
									Type identityElement)
{
	Type result = identityElement;
	uint2 globalId = {blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y};
	if(globalId.x < dimensions.x &&
	   globalId.y < dimensions.y)
	{
		result = tex2D<Type>(tIn,
							 static_cast<float>(globalId.x + offset.x) + 0.5f,
							 static_cast<float>(globalId.y + offset.y) + 0.5f);
	}

	BlockReduce2D<Type, F>(result, identityElement);
	if(threadIdx.x == 0 && threadIdx.y == 0)
		gOut[blockIdx.y * gridDim.x + blockIdx.x] = result;
}


template<class Type, ReduceFunc<Type> F, cudaMemcpyKind cpyKind = cudaMemcpyDeviceToDevice>
__host__ void KCReduceArray(Type& result,
							const Type* dData,
							size_t elementCount,
							Type identityElement,
							cudaStream_t stream = (cudaStream_t)0)
{
	static constexpr unsigned int TPB = StaticThreadPerBlock1D;
	static constexpr unsigned int SharedSize = (TPB / WarpSize) * sizeof(Type);

	unsigned int allocBig = (static_cast<unsigned int>(elementCount) + TPB - 1) / TPB;
	unsigned int allocSmall = (allocBig + TPB - 1) / TPB;
	DeviceMemory buffer(allocBig * sizeof(Type) +
						allocSmall * sizeof(Type));
	Type* dRead = static_cast<Type*>(buffer) + allocBig;
	Type* dWrite = static_cast<Type*>(buffer);

	unsigned int dataSize = static_cast<unsigned int>(elementCount);
	unsigned int gridSize;

	const Type* inData = dData;
	do
	{
		// Current Grid Size is reduced by previous data size
		gridSize = (dataSize + TPB - 1) / TPB;

		// KC Paralel Reduction
		ParalelReduction<Type, F> <<<gridSize, TPB, SharedSize, stream>>>
		(
			dWrite,
			inData,
			dataSize,
			identityElement
		);
		CUDA_KERNEL_CHECK();

		dataSize = gridSize;
		std::swap(dRead, dWrite);
		inData = dRead;
	} while(dataSize != 1);

	// Just get the data from gpu (first element at dRead
	CUDA_CHECK(cudaMemcpyAsync(&result, dRead, sizeof(Type), cpyKind, stream));
}

template<class Type, ReduceFunc<Type> F, cudaMemcpyKind cpyKind = cudaMemcpyDeviceToDevice>
__host__ void KCReduceTexture(Type& result,
							  cudaTextureObject_t texture,
							  const uint2& dim,
							  const uint2& offset,
							  Type identityElement,
							  cudaStream_t stream = (cudaStream_t)0)
{
	static constexpr Vector2ui TPB = StaticThreadPerBlock2D;
	static constexpr unsigned int SharedSize = (TPB[0] * TPB[1]);// / WarpSize) * sizeof(Type);

	dim3 blockSize = dim3(TPB[0], TPB[1]);
	dim3 gridSize;
	gridSize.x = (dim.x + TPB[0] - 1) / TPB[0];
	gridSize.y = (dim.y + TPB[1] - 1) / TPB[1];
	DeviceMemory reduceBuffer(gridSize.x * gridSize.y * sizeof(Type));
	Type* dReduceBuffer = static_cast<Type*>(reduceBuffer);

	// KC Paralel Reduction
	ParalelReductionTex<Type, F> <<<gridSize, blockSize, SharedSize, stream>>>
	(
		dReduceBuffer,
		texture,
		dim, offset,
		identityElement
	);
	CUDA_KERNEL_CHECK();

	// Array portion does the rest
	KCReduceArray<Type, F>(result,
						   dReduceBuffer,
						   gridSize.x * gridSize.y,
						   identityElement,
						   stream);
	CUDA_KERNEL_CHECK();
	// Just get the data from gpu (first element at dRead
	CUDA_CHECK(cudaMemcpyAsync(&result, dReduceBuffer, sizeof(Type), cpyKind, stream));
}

// Meta Definition
#define DEFINE_REDUCE_ARRAY_SINGLE(type, func, cpy) \
	template \
	__host__ void KCReduceArray<type, func, cpy>(type&, const type*, \
												 size_t, type, \
												 cudaStream_t);
#define DEFINE_REDUCE_TEXTURE_SINGLE(type, func, cpy) \
	template \
	__host__ void KCReduceTexture<type, func, cpy>(type&, cudaTextureObject_t, \
												   const uint2&, \
												   const uint2&, \
												   type, cudaStream_t);

// Cluster Definitions ARRAY
#define EXTERN_REDUCE_ARRAY_SINGLE(type, func, copy) \
	extern DEFINE_REDUCE_ARRAY_SINGLE(type, func, copy)

#define EXTERN_REDUCE_ARRAY_BOTH(type, func) \
	EXTERN_REDUCE_ARRAY_SINGLE(type, func, cudaMemcpyDeviceToHost) \
	EXTERN_REDUCE_ARRAY_SINGLE(type, func, cudaMemcpyDeviceToDevice)

#define EXTERN_REDUCE_ARRAY_ALL(type) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceAdd) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceSubtract) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceMultiply) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceDivide) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceMin) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceMax)

// Cluster Definitions TEXTURE
#define EXTERN_REDUCE_TEXTURE_SINGLE(type, func, copy) \
	extern DEFINE_REDUCE_TEXTURE_SINGLE(type, func, copy)

#define EXTERN_REDUCE_TEXTURE_BOTH(type, func) \
	EXTERN_REDUCE_TEXTURE_SINGLE(type, func, cudaMemcpyDeviceToHost) \
	EXTERN_REDUCE_TEXTURE_SINGLE(type, func, cudaMemcpyDeviceToDevice)

#define EXTERN_REDUCE_TEXTURE_ALL(type) \
	EXTERN_REDUCE_TEXTURE_BOTH(type, ReduceAdd) \
	EXTERN_REDUCE_TEXTURE_BOTH(type, ReduceSubtract) \
	EXTERN_REDUCE_TEXTURE_BOTH(type, ReduceMultiply) \
	EXTERN_REDUCE_TEXTURE_BOTH(type, ReduceDivide) \
	EXTERN_REDUCE_TEXTURE_BOTH(type, ReduceMin) \
	EXTERN_REDUCE_TEXTURE_BOTH(type, ReduceMax)

// Integral Types
EXTERN_REDUCE_ARRAY_ALL(int)
EXTERN_REDUCE_ARRAY_ALL(unsigned int)
EXTERN_REDUCE_ARRAY_ALL(float)
EXTERN_REDUCE_ARRAY_ALL(double)

// Vector Types
EXTERN_REDUCE_ARRAY_ALL(Vector2f)
EXTERN_REDUCE_ARRAY_ALL(Vector2d)
EXTERN_REDUCE_ARRAY_ALL(Vector2i)
EXTERN_REDUCE_ARRAY_ALL(Vector2ui)

EXTERN_REDUCE_ARRAY_ALL(Vector3f)
EXTERN_REDUCE_ARRAY_ALL(Vector3d)
EXTERN_REDUCE_ARRAY_ALL(Vector3i)
EXTERN_REDUCE_ARRAY_ALL(Vector3ui)

EXTERN_REDUCE_ARRAY_ALL(Vector4f)
EXTERN_REDUCE_ARRAY_ALL(Vector4d)
EXTERN_REDUCE_ARRAY_ALL(Vector4i)
EXTERN_REDUCE_ARRAY_ALL(Vector4ui)

// Matrix Types
EXTERN_REDUCE_ARRAY_ALL(Matrix2x2f)
EXTERN_REDUCE_ARRAY_ALL(Matrix2x2d)
EXTERN_REDUCE_ARRAY_ALL(Matrix2x2i)
EXTERN_REDUCE_ARRAY_ALL(Matrix2x2ui)

EXTERN_REDUCE_ARRAY_ALL(Matrix3x3f)
EXTERN_REDUCE_ARRAY_ALL(Matrix3x3d)
EXTERN_REDUCE_ARRAY_ALL(Matrix3x3i)
EXTERN_REDUCE_ARRAY_ALL(Matrix3x3ui)

EXTERN_REDUCE_ARRAY_ALL(Matrix4x4f)
EXTERN_REDUCE_ARRAY_ALL(Matrix4x4d)
EXTERN_REDUCE_ARRAY_ALL(Matrix4x4i)
EXTERN_REDUCE_ARRAY_ALL(Matrix4x4ui)

// Quaternion Types
EXTERN_REDUCE_ARRAY_BOTH(QuatF, ReduceMultiply)
EXTERN_REDUCE_ARRAY_BOTH(QuatD, ReduceMultiply)

// Texture Types
EXTERN_REDUCE_TEXTURE_ALL(float)
EXTERN_REDUCE_TEXTURE_ALL(float2)
EXTERN_REDUCE_TEXTURE_ALL(float4)