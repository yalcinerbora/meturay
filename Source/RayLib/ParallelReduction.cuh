#pragma once
/**
Parallel Reduction Meta Implementation
with Templates

Can define custom reduce function and custom type

Modern Paralel Reduction Code
Utilizing new Kepler warp data transfer discussed here
http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>
#include "DeviceMemory.h"
#include "CudaCheck.h"
#include "CudaConstants.h"
#include "Vector.cuh"
#include "Matrix.h"
#include "Quaternion.h"

template<class Type>
using ReduceFunc = Type (*)(const Type&, const Type&);

template <class Type, ReduceFunc<Type> F>
__device__ inline void WarpReduce(Type& val)
{
	static_assert(sizeof(Type) % sizeof(int) == 0, "Type should fit into integers perfectly.");

	#pragma unroll
	for(int offset = warpSize / 2; offset > 0; offset /= 2)
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
	static __shared__ Type shared[32];	// Shared mem 32 is max since 32*32 = 1024 threas
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
	static __shared__ Type shared[32];					// Shared mem 32 is max since 32*32 = 1024 threas
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
									uint2 offset,
									uint2 dimensions,
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
	if(threadIdx.x == 0 &&
	   threadIdx.y == 0)
		gOut[blockIdx.y * gridDim.x + blockIdx.x] = result;
}


template<class Type, ReduceFunc<Type> F, unsigned int TPB, cudaMemcpyKind cpyKind = cudaMemcpyDeviceToDevice>
__host__ void KCReduceArray(Type& result,
							const Type* dData,
							size_t offset, size_t size,
							Type identityElement,
							cudaStream_t stream = (cudaStream_t)0)
{
	unsigned int allocBig = (static_cast<unsigned int>(size) + TPB - 1) / TPB;
	unsigned int allocSmall = (allocBig + TPB - 1) / TPB;
	DeviceMemory dSwap1(allocBig * sizeof(Type)), dSwap2(allocSmall * sizeof(Type));
	Type* dRead = static_cast<Type*>(dSwap2);
	Type* dWrite = static_cast<Type*>(dSwap1);

	unsigned int dataSize = static_cast<unsigned int>(size);
	unsigned int gridSize;

	const Type* inData = dData + offset;
	do
	{
		// Current Grid Size is reduced by previous data size
		gridSize = (dataSize + TPB - 1) / TPB;

		// KC Paralel Reduction
		ParalelReduction<Type, F> <<<gridSize, TPB, 0, stream>>>
		(
			dWrite,
			inData,
			dataSize,
			identityElement
		);
		CUDA_KERNEL_CHECK();

		inData = dRead;
		dataSize = gridSize;
		std::swap(dRead, dWrite);
	} while(dataSize != 1);

	// Just get the data from gpu (first element at dRead
	CUDA_CHECK(cudaMemcpyAsync(&result, dRead, sizeof(Type), cpyKind, stream));
}

template<class Type, ReduceFunc<Type> F, unsigned int TPB_X, unsigned int TPB_Y, cudaMemcpyKind cpyKind = cudaMemcpyDeviceToDevice>
__host__ void KCReduceTexture(Type& result,
							  cudaTextureObject_t texture,
							  const uint2& dim,
							  Type identityElement,
							  cudaStream_t stream = (cudaStream_t)0)
{
	dim3 gridSize;
	gridSize.x = (static_cast<unsigned int>(dim.x) + TPB_X - 1) / TPB_X;
	gridSize.y = (static_cast<unsigned int>(dim.y) + TPB_Y - 1) / TPB_Y;
	DeviceMemory reduceBuffer(gridSize.x * gridSize.y * sizeof(Type));
	Type* dReduceBuffer = static_cast<Type*>(reduceBuffer);
	

	uint2 offset = {0,0};
	// KC Paralel Reduction
	ParalelReductionTex<Type, F> <<<gridSize, dim3(TPB_X, TPB_Y), 0, stream>>>
	(
		dReduceBuffer,
		texture,
		offset,
		dim,
		identityElement
	);
	CUDA_KERNEL_CHECK();

	// Array portion does the rest
	KCReduceArray<Type, F, TPB_X * TPB_Y>(result,
										  dReduceBuffer,
										  0,
										  gridSize.x * gridSize.y,
										  identityElement,
										  stream);
	CUDA_KERNEL_CHECK();
	// Just get the data from gpu (first element at dRead
	CUDA_CHECK(cudaMemcpyAsync(&result, dReduceBuffer, sizeof(Type), cpyKind, stream));
}

// Some Basic Implementations in order to prevent duplicates
template <class T>
inline __device__ T ReduceAdd(const T& a, const T&b)
{
	return a + b;
}

template <class T>
inline __device__ T ReduceSubtract(const T& a, const T&b)
{
	return a - b;
}

template <class T>
inline __device__ T ReduceMultiply(const T& a, const T&b)
{
	return a * b;
}

template <class T>
inline __device__ T ReduceDivide(const T& a, const T&b)
{
	return a / b;
}

template<class T>
using EnableArithmetic = typename std::enable_if<std::is_arithmetic<T>::value, T>::type;

template<class T>
using EnableVectorOrMatrix = typename std::enable_if<IsVectorType<T>::value || 
													 IsMatrixType<T>::value, T>::type;

template<class T, class = ArithmeticEnable<T>>
inline __device__  EnableArithmetic<T> ReduceMin(const T& a, const T&b)
{
	return std::min(a, b);
}

template<class T, class = ArithmeticEnable<T>>
inline __device__ EnableArithmetic<T> ReduceMax(const T& a, const T&b)
{
	return std::max(a, b);
}

template<class T>
inline __device__  EnableVectorOrMatrix<T> ReduceMin(const T& a, const T&b)
{
	return T::Min(a, b);
}

template<class T>
inline __device__  EnableVectorOrMatrix<T> ReduceMax(const T& a, const T&b)
{
	return T::Max(a, b);
}

// Meta Definition
#define DEFINE_REDUCE_ARRAY_SINGLE(type, func, cpy) \
	template \
	__host__ void KCReduceArray<type, func, StaticThreadPerBlock1D, cpy>(type&, const type*, \
																		 size_t, size_t, \
																		 type, cudaStream_t);
#define DEFINE_REDUCE_TEXTURE_SINGLE(type, func, cpy) \
	template \
	__host__ void KCReduceTexture<type, func, StaticThreadPerBlock2D_X, StaticThreadPerBlock2D_Y, cpy>(type&, cudaTextureObject_t, \
																									   const uint2& dim, \
																									   type, cudaStream_t);

// Cluster Definitions ARRAY
#define EXTERN_REDUCE_ARRAY_SINGLE(type, func, copy) \
	extern DEFINE_REDUCE_ARRAY_SINGLE(type, func, copy)

#define EXTERN_REDUCE_ARRAY_BOTH(type, func) \
	EXTERN_REDUCE_ARRAY_SINGLE(type, func, cudaMemcpyDeviceToHost) \
	EXTERN_REDUCE_ARRAY_SINGLE(type, func, cudaMemcpyDeviceToDevice)
	
#define EXTERN_REDUCE_ARRAY_ALL(type) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceAdd<type>) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceSubtract<type>) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceMultiply<type>) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceDivide<type>) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceMin<type>) \
	EXTERN_REDUCE_ARRAY_BOTH(type, ReduceMax<type>)

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

//// Texture Types
//EXTERN_REDUCE_TEXTURE_ALL(float)
//EXTERN_REDUCE_TEXTURE_ALL(float2)
//EXTERN_REDUCE_TEXTURE_ALL(float4)