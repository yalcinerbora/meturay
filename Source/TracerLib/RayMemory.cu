#include "RayMemory.h"

#include <cub/cub.cuh>
#include <type_traits>

#include "RayLib/Log.h"

#include "CudaConstants.h"
#include "TracerDebug.h"
#include "HitFunctions.cuh"

static constexpr uint32_t INVALID_LOCATION = std::numeric_limits<uint32_t>::max();

struct ValidSplit
{
	__device__ __host__
	__forceinline__ bool operator()(const uint32_t &ids) const
	{
		return (ids != INVALID_LOCATION);
	}
};

__global__ void FillMatIdsForSort(HitKey* gKeys, RayId* gIds,
								  const HitKey* gMaterialHits,
								  uint32_t rayCount)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount;
		globalId += blockDim.x * gridDim.x)
	{
		gKeys[globalId] = gMaterialHits[globalId];
		gIds[globalId] = globalId;
	}
}

__global__ void ResetHitIds(HitKey* gAcceleratorKeys, RayId* gIds,
							HitKey* gMaterialKeys, const RayGMem* gRays,
							uint32_t rayCount)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount;
		globalId += blockDim.x * gridDim.x)
	{
		HitKey initalKey = HitKey::BoundaryMatKey;
		if(gRays[globalId].tMin == INFINITY)
		{
			initalKey = HitKey::InvalidKey;
		}

		gIds[globalId] = globalId;
		gAcceleratorKeys[globalId] = HitKey::InvalidKey;
		gMaterialKeys[globalId] = initalKey;
	}
}

__global__ void FindSplitsSparse(uint32_t* gPartLoc,
								 const HitKey* gKeys,
								 const uint32_t locCount)
{
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < locCount;
		globalId += blockDim.x * gridDim.x)
	{
		HitKey key = gKeys[globalId];
		HitKey keyN = gKeys[globalId + 1];

		uint16_t keyBatch = HitKey::FetchBatchPortion(key);
		uint16_t keyNBatch = HitKey::FetchBatchPortion(keyN);

		// Write location if split is found
		if(keyBatch != keyNBatch) gPartLoc[globalId + 1] = globalId + 1;
		else gPartLoc[globalId + 1] = INVALID_LOCATION;
	}

	// Init first location also
	if((blockIdx.x * blockDim.x + threadIdx.x) == 0)
		gPartLoc[0] = 0;
}

__global__ void FindSplitBatches(uint16_t* gBatches,
								 const uint32_t* gDenseIds,
								 const HitKey* gSparseKeys,
								 const uint32_t locCount)
{
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < locCount;
		globalId += blockDim.x * gridDim.x)
	{
		uint32_t index = gDenseIds[globalId];
		HitKey key = gSparseKeys[index];
		gBatches[globalId] = HitKey::FetchBatchPortion(key);
	}
}

void RayMemory::ResizeRayMemory(RayGMem*& dRays, void*& dRayAxData,
								DeviceMemory& mem,
								size_t rayCount,
								size_t perRayAuxSize)
{
	size_t sizeOfRays = rayCount * sizeof(RayGMem);
	size_t sizeOfAuxiliary = rayCount * perRayAuxSize;
	sizeOfAuxiliary = AlignByteCount * ((sizeOfAuxiliary + (AlignByteCount - 1)) / AlignByteCount);

	size_t requiredSize = sizeOfAuxiliary + sizeOfRays;
	if(mem.Size() < requiredSize)
		mem = std::move(DeviceMemory(requiredSize));

	size_t offset = 0;
	std::uint8_t* dRay = static_cast<uint8_t*>(mem);
	dRays = reinterpret_cast<RayGMem*>(dRay + offset);
	offset += sizeOfRays;
	dRayAxData = reinterpret_cast<void*>(dRay + offset);
	offset += sizeOfAuxiliary;
	assert(requiredSize == offset);
}

RayMemory::RayMemory()
{}

void RayMemory::SwapRays()
{
	DeviceMemory temp = std::move(memIn);
	memIn = std::move(memOut);
	memOut = std::move(temp);

	RayGMem* temp0;
	void* temp1;

	temp0 = dRayIn;
	dRayIn = dRayOut;
	dRayOut = temp0;

	temp1 = dRayAuxIn;
	dRayAuxIn = dRayAuxOut;
	dRayAuxOut = temp1;
}

void RayMemory::ResetHitMemory(size_t rayCount, size_t hitStructSize)
{
	// Align to proper memory strides
	size_t sizeOfMaterialKeys = sizeof(HitKey) * rayCount;
	sizeOfMaterialKeys = AlignByteCount * ((sizeOfMaterialKeys + (AlignByteCount - 1)) / AlignByteCount);

	size_t sizeOfTransformIds = sizeof(TransformId) * rayCount;
	sizeOfTransformIds = AlignByteCount * ((sizeOfTransformIds + (AlignByteCount - 1)) / AlignByteCount);

	size_t sizeOfPrimitiveIds = sizeof(PrimitiveId) * rayCount;
	sizeOfPrimitiveIds = AlignByteCount * ((sizeOfPrimitiveIds + (AlignByteCount - 1)) / AlignByteCount);

	size_t sizeOfHitStructs = hitStructSize * rayCount;
	sizeOfHitStructs = AlignByteCount * ((sizeOfHitStructs + (AlignByteCount - 1)) / AlignByteCount);
	//
	//
	size_t sizeOfIds = sizeof(RayId) * rayCount;
	sizeOfIds = AlignByteCount * ((sizeOfIds + (AlignByteCount - 1)) / AlignByteCount);

	size_t sizeOfAcceleratorKeys = sizeof(HitKey) * rayCount;
	sizeOfAcceleratorKeys = AlignByteCount * ((sizeOfAcceleratorKeys + (AlignByteCount - 1)) / AlignByteCount);

	// Find out sort auxiliary storage
	cub::DoubleBuffer<HitKey::Type> dbKeys(nullptr, nullptr);
	cub::DoubleBuffer<RayId> dbIds(nullptr, nullptr);
	CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, cubSortMemSize,
											   dbKeys, dbIds,
											   static_cast<int>(rayCount)));


	// Check if while partitioning  double buffer data is
	// enough for using (Unique and Scan) algos
	uint32_t* in = nullptr;
	uint32_t* out = nullptr;
	uint32_t* count = nullptr;
	CUDA_CHECK(cub::DeviceSelect::If(nullptr, cubIfMemSize,
									 in, out, count,
									 static_cast<int>(rayCount),
									 ValidSplit()));

	// Select algo reads from split locations and writes to backbuffer Ids (half is used)
	// uses backbuffer ids other half as auxiliary buffer
	// This code tries to increase it accordingly
	// Output Count of If also should be considered (add sizeof uint32_t)
	size_t sizeOfTempMemory = std::max(cubSortMemSize, cubIfMemSize + sizeof(uint32_t));
	sizeOfTempMemory = (AlignByteCount * ((sizeOfTempMemory + (AlignByteCount - 1)) / AlignByteCount));

	// Finally allocate
	size_t requiredSize = ((sizeOfIds + sizeOfAcceleratorKeys) * 2 +
						   sizeOfMaterialKeys +
						   sizeOfTransformIds +
						   sizeOfPrimitiveIds +
						   sizeOfHitStructs +
						   sizeOfTempMemory);

	// Reallocate if memory is not enough
	if(memHit.Size() < requiredSize)
		memHit = std::move(DeviceMemory(requiredSize));

	// Populate pointers
	size_t offset = 0;
	std::uint8_t* dBasePtr = static_cast<uint8_t*>(memHit);
	dMaterialKeys = reinterpret_cast<HitKey*>(dBasePtr + offset);
	offset += sizeOfMaterialKeys;
	dTransformIds = reinterpret_cast<TransformId*>(dBasePtr + offset);
	offset += sizeOfTransformIds;
	dPrimitiveIds = reinterpret_cast<PrimitiveId*>(dBasePtr + offset);
	offset += sizeOfPrimitiveIds;
	dHitStructs = HitStructPtr(reinterpret_cast<void*>(dBasePtr + offset), static_cast<int>(hitStructSize));
	offset += sizeOfHitStructs;
	dIds0 = reinterpret_cast<RayId*>(dBasePtr + offset);
	offset += sizeOfIds;
	dKeys0 = reinterpret_cast<HitKey*>(dBasePtr + offset);
	offset += sizeOfAcceleratorKeys;
	dIds1 = reinterpret_cast<RayId*>(dBasePtr + offset);
	offset += sizeOfIds;
	dKeys1 = reinterpret_cast<HitKey*>(dBasePtr + offset);
	offset += sizeOfAcceleratorKeys;
	dTempMemory = reinterpret_cast<void*>(dBasePtr + offset);
	offset += sizeOfTempMemory;
	assert(requiredSize == offset);

	dCurrentIds = dIds0;
	dCurrentKeys = dKeys0;

	// Make nullptr if no hitstruct is needed
	if(sizeOfHitStructs == 0)
		dHitStructs = HitStructPtr(nullptr, static_cast<int>(hitStructSize));

	// Initialize memory
	CudaSystem::GridStrideKC_X(leaderDeviceId, 0, 0, rayCount,
							   ResetHitIds,
							   dCurrentKeys, dCurrentIds, dMaterialKeys, dRayIn,
							   static_cast<uint32_t>(rayCount));
}

void RayMemory::SortKeys(RayId*& ids, HitKey*& keys,
						 size_t count,
						 const Vector2i& bitMaxValues)
{
	// Sort Call over buffers
	HitKey* keysOther = (dCurrentKeys == dKeys0) ? dKeys1 : dKeys0;
	RayId* idsOther = (dCurrentIds == dIds0) ? dIds1 : dIds0;
	cub::DoubleBuffer<HitKey::Type> dbKeys(reinterpret_cast<HitKey::Type*>(dCurrentKeys),
										   reinterpret_cast<HitKey::Type*>(keysOther));
	cub::DoubleBuffer<RayId> dbIds(dCurrentIds,
								   idsOther);
	int bitStart = 0;
	int bitEnd = bitMaxValues[1];

	CUDA_CHECK(cudaDeviceSynchronize());

	// First sort internals
	if(bitStart != bitEnd)
	{
		CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dTempMemory, cubSortMemSize,
												   dbKeys, dbIds,
												   static_cast<int>(count),
												   bitStart, bitEnd,
												   (cudaStream_t)0,
												   METU_DEBUG_BOOL));
	}

	// Then sort batches
	bitStart = HitKey::IdBits;
	bitEnd = HitKey::IdBits + bitMaxValues[0];
	if(bitStart != bitEnd)
	{
		CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dTempMemory, cubSortMemSize,
												   dbKeys, dbIds,
												   static_cast<int>(count),
												   bitStart, bitEnd,
												   (cudaStream_t)0,
												   METU_DEBUG_BOOL));
	}

	ids = dbIds.Current();
	keys = reinterpret_cast<HitKey*>(dbKeys.Current());
	dCurrentIds = ids;
	dCurrentKeys = keys;
}

RayPartitions<uint32_t> RayMemory::Partition(uint32_t rayCount)
{
	// Use double buffers for partition auxilary data
	RayId* dEmptyIds = (dCurrentIds == dIds0) ? dIds1 : dIds0;
	HitKey* dEmptyKeys = (dCurrentKeys == dKeys0) ? dKeys1 : dKeys0;

	// Generate Names that make sense for the operation
	// We have total of three buffers
	// Temp Memory will be used for temp memory
	// (it holds enough space for both sort and select)
	//
	// dSparseSplitIndices (a.k.a. dEmptyKeys)
	// will be used as intermediate buffer
	uint32_t* dSparseSplitIndices = reinterpret_cast<uint32_t*>(dEmptyKeys);
	uint32_t* dDenseSplitIndices = reinterpret_cast<uint32_t*>(dEmptyIds);
	uint32_t* dSelectCount = static_cast<uint32_t*>(dTempMemory);
	void* dSelectTempMemory = dSelectCount + 1;

	// Find Split Locations
	// Read from dKeys -> dEmptyKeys
	uint32_t locCount = rayCount - 1;
	CudaSystem::GridStrideKC_X(leaderDeviceId, 0, 0, rayCount,
							   FindSplitsSparse,
							   dSparseSplitIndices, dCurrentKeys, locCount);

	// Make Splits Dense
	// From dEmptyKeys -> dEmptyIds
	CUDA_CHECK(cub::DeviceSelect::If(dSelectTempMemory, cubIfMemSize,
									 dSparseSplitIndices, dDenseSplitIndices, dSelectCount,
									 static_cast<int>(rayCount),
									 ValidSplit(),
									 (cudaStream_t)0,
									 METU_DEBUG_BOOL));

	// Copy Reduced Count
	uint32_t hSelectCount;
	CUDA_CHECK(cudaMemcpy(&hSelectCount, dSelectCount,
						  sizeof(uint32_t), cudaMemcpyDeviceToHost));

	// Find The Hit Keys for each split
	// From dEmptyIds, dKeys -> dEmptyKeys
	uint16_t* dBatches = reinterpret_cast<uint16_t*>(dSparseSplitIndices);
	CudaSystem::GridStrideKC_X(leaderDeviceId, 0, 0, rayCount,
							   FindSplitBatches,
							   dBatches,
							   dDenseSplitIndices,
							   dCurrentKeys,
							   hSelectCount);

	// We need to get dDenseIndices & dDenseKeys
	// Memcopy to vectors
	std::vector<uint16_t> hDenseKeys(hSelectCount);
	std::vector<uint32_t> hDenseIndices(hSelectCount);
	CUDA_CHECK(cudaMemcpy(hDenseKeys.data(), dBatches,
						  sizeof(uint16_t) * hSelectCount,
						  cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(hDenseIndices.data(), dDenseSplitIndices,
						  sizeof(uint32_t) * hSelectCount,
						  cudaMemcpyDeviceToHost));

	// Construct The Set
	// Add extra index to end as rayCount for cleaner code
	hDenseIndices.push_back(rayCount);
	RayPartitions<uint32_t> partitions;
	for(uint32_t i = 0; i < hSelectCount; i++)
	{
		uint32_t id = hDenseKeys[i];
		uint32_t offset = hDenseIndices[i];
		size_t count = hDenseIndices[i + 1] - hDenseIndices[i];
		partitions.emplace(ArrayPortion<uint32_t>{id, offset, count});
	}
	// Done!
	return std::move(partitions);
}

void RayMemory::FillRayIdsForSort(uint32_t rayCount)
{
	CudaSystem::GridStrideKC_X(leaderDeviceId, 0, 0, rayCount,
							   FillMatIdsForSort,
							   dCurrentKeys, dCurrentIds, dMaterialKeys,
							   static_cast<uint32_t>(rayCount));
}