#pragma once
/**
Bitonic Sort Meta Implementation
with custom Key/Value pairs

Can define custom types and comparison operators
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include "ComparisonFunctions.cuh"

// Out ones
template <class Type, class Key, CompFunc<Key> Comp>
__host__ void BitonicSort(Type* data, Key* key,
						  size_t elementCount,
						  cudaStream_t stream = (cudaStream_t)0)
{
}

template <class Type, class Key, CompFunc<Key> Comp>
__host__ void BitonicSort(Type* data,  Key* key,
						  const Type* data, const Key* key,
						  size_t elementCount,
						  cudaStream_t stream = (cudaStream_t)0)
{
}

// Inplace Ones
template <class Type, CompFunc<Key> Comp>
__host__ void BitonicSort(Type* data, size_t elementCount,
						  cudaStream_t stream = (cudaStream_t)0)
{
}

template <class Type, class Key, CompFunc<Key> Comp>
__host__ void BitonicSort(Type* data, Key* key, size_t elementCount,
						  cudaStream_t stream = (cudaStream_t)0)
{
}

// Meta Definitions
#define DEFINE_BITONIC_VALUE(type, key, comp) \
	template \
	__host__ void BitonicSort<type, comp>(type*, const type*, \
											   size_t, \
											   cudaStream_t);

#define DEFINE_BITONIC_VALUE_INPLACE(type, comp) \
	__host__ void BitonicSort<type, key, comp>(type*, const type*, \
										   size_t, int, int, \
										   cudaStream_t);

#define DEFINE_BITONIC_KEY(type, key, comp) \
	__host__ void BitonicSort<type, comp>(type*, const type*, \
										  size_t, int, int, \
										  cudaStream_t);

#define DEFINE_BITONIC_KEY_INPLACE(type, key, comp) \
	__host__ void BitonicSort<type, comp>(type*, const type*, \
										  size_t, int, int, \
										  cudaStream_t);

#define DEFINE_BITONIC_KEY_VALUE(key, type, order) \
	template \
	__host__ void KCRadixSortArray<type, key, order>(type*, key*, \
													 const type*, const key*, \
													 size_t, int, int, \
													 cudaStream_t);

#define DEFINE_RADIX_VALUE_BOTH(type) \
	DEFINE_RADIX_VALUE(type, true); \
	DEFINE_RADIX_VALUE(type, false);

#define DEFINE_RADIX_KEY_VALUE_BOTH(key, type) \
	DEFINE_RADIX_KEY_VALUE(key, type, true); \
	DEFINE_RADIX_KEY_VALUE(key, type, false);

#define EXTERN_RADIX_VALUE_BOTH(type) \
	extern DEFINE_RADIX_VALUE(type, true); \
	extern DEFINE_RADIX_VALUE(type, false);

#define EXTERN_RADIX_KEY_VALUE_BOTH(key, type) \
	extern DEFINE_RADIX_KEY_VALUE(key, type, true); \
	extern DEFINE_RADIX_KEY_VALUE(key, type, false);

#define EXTERN_RADIX_KEY_VALUE_ALL(type) \
	EXTERN_RADIX_KEY_VALUE_BOTH(int, type) \
	EXTERN_RADIX_KEY_VALUE_BOTH(unsigned int, type) \
	EXTERN_RADIX_KEY_VALUE_BOTH(float, type) \
	EXTERN_RADIX_KEY_VALUE_BOTH(double, type) \
	EXTERN_RADIX_KEY_VALUE_BOTH(int64_t, type) \
	EXTERN_RADIX_KEY_VALUE_BOTH(uint64_t, type)

#define DEFINE_RADIX_KEY_VALUE_ALL(type) \
	DEFINE_RADIX_KEY_VALUE_BOTH(int, type) \
	DEFINE_RADIX_KEY_VALUE_BOTH(unsigned int, type) \
	DEFINE_RADIX_KEY_VALUE_BOTH(float, type) \
	DEFINE_RADIX_KEY_VALUE_BOTH(double, type) \
	DEFINE_RADIX_KEY_VALUE_BOTH(int64_t, type) \
	DEFINE_RADIX_KEY_VALUE_BOTH(uint64_t, type)

//// Meta Definitions
//#define DEFINE_RADIX_OUT(type, func) \
//	template \
//	__host__ void KCExclusiveScanArray<type, key>(type*, const type*, \
//												  size_t, \
//												  cudaStream_t);
//
//#define DEFINE_RADIX_INOUT(type, func) \
//	template \
//	__host__ void KCRadixSortArray<type, func>(type*, size_t, \
//											   cudaStream_t);
//
//#define DEFINE_RADIX_BOTH(type, func) \
//	DEFINE_RADIX_OUT(type, func) \
//	DEFINE_RADIX_INOUT(type, func)
//
//#define DEFINE_RADIX_ALL(type) \
//	DEFINE_RADIX_BOTH(type, ReduceST) \
//	DEFINE_RADIX_BOTH(type, ReduceGT)
//
//// Extern Definitions
//#define EXTERN_RADIX_BOTH(type, func) \
//	extern DEFINE_RADIX_OUT(type, func) \
//	extern DEFINE_RADIX_INOUT(type, func)
//
//#define EXTERN_RADIX_ALL(type) \
//	EXTERN_RADIX_BOTH(type, ReduceST) \
//	EXTERN_RADIX_BOTH(type, ReduceGT)

//void BitonicSort::Sort(
//	ID3D12GraphicsCommandList *pCommandList,
//	D3D12_GPU_VIRTUAL_ADDRESS SortKeyBuffer,
//	D3D12_GPU_VIRTUAL_ADDRESS IndexBuffer,
//	UINT ElementCount,
//	bool IsPartiallyPreSorted,
//	bool SortAscending)
//{
//	if(ElementCount == 0) return;
//
//	const uint32_t AlignedNumElements = AlignPowerOfTwo(ElementCount);
//	const uint32_t MaxIterations = Log2(std::max(2048u, AlignedNumElements)) - 10;
//
//	pCommandList->SetComputeRootSignature(m_pRootSignature);
//
//	struct InputConstants
//	{
//		UINT NullIndex;
//		UINT ListCount;
//	};
//	InputConstants constants{SortAscending ? 0xffffffff : 0, ElementCount};
//	pCommandList->SetComputeRoot32BitConstants(GenericConstants, SizeOfInUint32(InputConstants), &constants, 0);
//
//	// Generate execute indirect arguments
//	pCommandList->SetPipelineState(m_pBitonicIndirectArgsCS);
//
//	auto argToUAVTransition = CD3DX12_RESOURCE_BARRIER::Transition(m_pDispatchArgs, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
//	pCommandList->ResourceBarrier(1, &argToUAVTransition);
//
//	pCommandList->SetComputeRoot32BitConstant(ShaderSpecificConstants, MaxIterations, 0);
//	pCommandList->SetComputeRootUnorderedAccessView(OutputUAV, m_pDispatchArgs->GetGPUVirtualAddress());
//	pCommandList->SetComputeRootUnorderedAccessView(IndexBufferUAV, IndexBuffer);
//	pCommandList->Dispatch(1, 1, 1);
//
//	// Pre-Sort the buffer up to k = 2048.  This also pads the list with invalid indices
//	// that will drift to the end of the sorted list.
//	auto argToIndirectArgTransition = CD3DX12_RESOURCE_BARRIER::Transition(m_pDispatchArgs, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
//	pCommandList->ResourceBarrier(1, &argToIndirectArgTransition);
//	pCommandList->SetComputeRootUnorderedAccessView(OutputUAV, SortKeyBuffer);
//
//	auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
//	if(!IsPartiallyPreSorted)
//	{
//		pCommandList->SetPipelineState(m_pBitonicPreSortCS);
//		pCommandList->ExecuteIndirect(m_pCommandSignature, 1, m_pDispatchArgs, 0, nullptr, 0);
//		pCommandList->ResourceBarrier(1, &uavBarrier);
//	}
//
//	uint32_t IndirectArgsOffset = cIndirectArgStride;
//
//	// We have already pre-sorted up through k = 2048 when first writing our list, so
//	// we continue sorting with k = 4096.  For unnecessarily large values of k, these
//	// indirect dispatches will be skipped over with thread counts of 0.
//
//	for(uint32_t k = 4096; k <= AlignedNumElements; k *= 2)
//	{
//		pCommandList->SetPipelineState(m_pBitonicOuterSortCS);
//
//		for(uint32_t j = k / 2; j >= 2048; j /= 2)
//		{
//			struct OuterSortConstants
//			{
//				UINT k;
//				UINT j;
//			} constants{k, j};
//
//			pCommandList->SetComputeRoot32BitConstants(ShaderSpecificConstants, SizeOfInUint32(OuterSortConstants), &constants, 0);
//			pCommandList->ExecuteIndirect(m_pCommandSignature, 1, m_pDispatchArgs, IndirectArgsOffset, nullptr, 0);
//			pCommandList->ResourceBarrier(1, &uavBarrier);
//			IndirectArgsOffset += cIndirectArgStride;
//		}
//
//		pCommandList->SetPipelineState(m_pBitonicInnerSortCS);
//		pCommandList->ExecuteIndirect(m_pCommandSignature, 1, m_pDispatchArgs, IndirectArgsOffset, nullptr, 0);
//		pCommandList->ResourceBarrier(1, &uavBarrier);
//		IndirectArgsOffset += cIndirectArgStride;
//	}
//}