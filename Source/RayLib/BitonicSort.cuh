#pragma once
/**
Bitonic Sort Meta Implementation
with custom Key/Value pairs

Can define custom types and comparison operators
*/

template<class Type>
using Comparator = bool(*)(const Type&, const Type&);

template <class Type, Comparator<Type> C,  unsigned int TPB>
__host__ void BitonicSort(T*,
						  size_t elementCount,
						  size_t offset,)

void BitonicSort::Sort(
	ID3D12GraphicsCommandList *pCommandList,
	D3D12_GPU_VIRTUAL_ADDRESS SortKeyBuffer,
	D3D12_GPU_VIRTUAL_ADDRESS IndexBuffer,
	UINT ElementCount,
	bool IsPartiallyPreSorted,
	bool SortAscending)
{
	if(ElementCount == 0) return;

	const uint32_t AlignedNumElements = AlignPowerOfTwo(ElementCount);
	const uint32_t MaxIterations = Log2(std::max(2048u, AlignedNumElements)) - 10;

	pCommandList->SetComputeRootSignature(m_pRootSignature);

	struct InputConstants
	{
		UINT NullIndex;
		UINT ListCount;
	};
	InputConstants constants{SortAscending ? 0xffffffff : 0, ElementCount};
	pCommandList->SetComputeRoot32BitConstants(GenericConstants, SizeOfInUint32(InputConstants), &constants, 0);

	// Generate execute indirect arguments
	pCommandList->SetPipelineState(m_pBitonicIndirectArgsCS);

	auto argToUAVTransition = CD3DX12_RESOURCE_BARRIER::Transition(m_pDispatchArgs, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	pCommandList->ResourceBarrier(1, &argToUAVTransition);

	pCommandList->SetComputeRoot32BitConstant(ShaderSpecificConstants, MaxIterations, 0);
	pCommandList->SetComputeRootUnorderedAccessView(OutputUAV, m_pDispatchArgs->GetGPUVirtualAddress());
	pCommandList->SetComputeRootUnorderedAccessView(IndexBufferUAV, IndexBuffer);
	pCommandList->Dispatch(1, 1, 1);

	// Pre-Sort the buffer up to k = 2048.  This also pads the list with invalid indices
	// that will drift to the end of the sorted list.
	auto argToIndirectArgTransition = CD3DX12_RESOURCE_BARRIER::Transition(m_pDispatchArgs, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
	pCommandList->ResourceBarrier(1, &argToIndirectArgTransition);
	pCommandList->SetComputeRootUnorderedAccessView(OutputUAV, SortKeyBuffer);

	auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
	if(!IsPartiallyPreSorted)
	{
		pCommandList->SetPipelineState(m_pBitonicPreSortCS);
		pCommandList->ExecuteIndirect(m_pCommandSignature, 1, m_pDispatchArgs, 0, nullptr, 0);
		pCommandList->ResourceBarrier(1, &uavBarrier);
	}

	uint32_t IndirectArgsOffset = cIndirectArgStride;

	// We have already pre-sorted up through k = 2048 when first writing our list, so
	// we continue sorting with k = 4096.  For unnecessarily large values of k, these
	// indirect dispatches will be skipped over with thread counts of 0.

	for(uint32_t k = 4096; k <= AlignedNumElements; k *= 2)
	{
		pCommandList->SetPipelineState(m_pBitonicOuterSortCS);

		for(uint32_t j = k / 2; j >= 2048; j /= 2)
		{
			struct OuterSortConstants
			{
				UINT k;
				UINT j;
			} constants{k, j};

			pCommandList->SetComputeRoot32BitConstants(ShaderSpecificConstants, SizeOfInUint32(OuterSortConstants), &constants, 0);
			pCommandList->ExecuteIndirect(m_pCommandSignature, 1, m_pDispatchArgs, IndirectArgsOffset, nullptr, 0);
			pCommandList->ResourceBarrier(1, &uavBarrier);
			IndirectArgsOffset += cIndirectArgStride;
		}

		pCommandList->SetPipelineState(m_pBitonicInnerSortCS);
		pCommandList->ExecuteIndirect(m_pCommandSignature, 1, m_pDispatchArgs, IndirectArgsOffset, nullptr, 0);
		pCommandList->ResourceBarrier(1, &uavBarrier);
		IndirectArgsOffset += cIndirectArgStride;
	}
}