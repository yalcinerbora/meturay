#include "RayMemory.h"
#include "RayLib/CudaConstants.h"
#include <cub/cub.cuh>
#include <cstddef>
#include <type_traits>

struct ValidSplit
{
	__device__ __host__ 
	__forceinline__ bool operator()(const uint32_t &ids) const
	{
		return (ids != RayMemory::InvalidData);
	}
};

__global__ void FillHitIdsForSort(HitKey* gKeys, RayId* gIds,
								  const HitGMem* gHits, 
								  uint32_t rayCount)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount;
		globalId += blockDim.x * gridDim.x)
	{
		gKeys[globalId] = gHits[globalId].hitKey;
		gIds[globalId] = globalId;
	}
}

__global__ void ResetHitIds(HitKey* gKeys, RayId* gIds, 
							HitGMem* gHits, uint32_t rayCount)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; 
		globalId += blockDim.x * gridDim.x)
	{
		gIds[globalId] = globalId;
		gKeys[globalId] = RayMemory::InvalidKey;
		gHits[globalId].hitKey = RayMemory::InvalidKey;
		gHits[globalId].innerId = RayMemory::InvalidData;
	}
}

__global__ void FindSplitsSparse(uint32_t* gPartLoc,
								 const HitKey* gKeys,
								 const uint32_t locCount,
								 const Vector2i bitRange)
{
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < locCount;
		globalId += blockDim.x * gridDim.x)
	{
		HitKey key = gKeys[globalId];
		HitKey keyN = gKeys[globalId + 1];

		key >>= bitRange[0];
		key &= ((0x1 << (bitRange[1] - bitRange[0])) - 1);

		keyN >>= bitRange[0];
		keyN &= ((0x1 << (bitRange[1] - bitRange[0])) - 1);

		// Write location if split is found
		if(key != keyN) gPartLoc[globalId + 1] = globalId;
		else gPartLoc[globalId + 1] = RayMemory::InvalidData;
	}

	if((blockIdx.x * blockDim.x + threadIdx.x) == 0)
		gPartLoc[0] = 0;
}

__global__ void FindSplitKeys(HitKey* gDenseKeys,
							  const uint32_t* gDenseIds,
							  const HitKey* gSparseKeys,
							  const uint32_t locCount,
							  const Vector2i bitRange)
{
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < locCount;
		globalId += blockDim.x * gridDim.x)
	{
		uint32_t index = gDenseIds[globalId];
		HitKey key = gSparseKeys[index];
		key >>= bitRange[0];
		key &= ((0x1 << (bitRange[1] - bitRange[0])) - 1);

		gDenseKeys[globalId] = key;
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

void RayMemory::ResetHitMemory(size_t rayCount)
{
	// Align to proper memory strides
	size_t sizeOfKeys = sizeof(HitKey) * rayCount;
	sizeOfKeys = AlignByteCount * ((sizeOfKeys + (AlignByteCount - 1)) / AlignByteCount);

	size_t sizeOfIds = sizeof(RayId) * rayCount;
	sizeOfIds = AlignByteCount * ((sizeOfIds + (AlignByteCount - 1)) / AlignByteCount);

	size_t sizeOfHits = sizeof(HitGMem) * rayCount;
	sizeOfHits = AlignByteCount * ((sizeOfHits + (AlignByteCount - 1)) / AlignByteCount);

	// Find out sort auxiliary storage
	cub::DoubleBuffer<HitKey> dbKeys;
	cub::DoubleBuffer<RayId> dbIds;
	size_t sizeOfTempMemory = 0;
	CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sizeOfTempMemory,
											   dbKeys, dbIds,
											   static_cast<int>(rayCount)));
	

	// Check if while partitioning  double buffer data is 
	// enough for using (Unique and Scan) algos	
	size_t selectTempMemorySize = 0;
	uint32_t* in = nullptr;
	uint32_t* out = nullptr;
	uint32_t* count = nullptr;
	CUDA_CHECK(cub::DeviceSelect::If(nullptr, selectTempMemorySize,
									 in, out, count,
									 static_cast<int>(rayCount),
									 ValidSplit()));
	// Output Count of If also should be considered
	selectTempMemorySize += sizeof(uint32_t);

	// Select algo reads from split locations and writes to backbuffer Ids (half is used)
	// uses backbuffer ids other half as auxiliary buffer
	// This code tries to increase it accordingly
	if(selectTempMemorySize > sizeOfTempMemory)
		sizeOfTempMemory = (AlignByteCount * ((selectTempMemorySize + (AlignByteCount - 1)) / AlignByteCount));

	// Finally allocate
	size_t requiredSize = (sizeOfIds + sizeOfKeys) * 2 + sizeOfHits + sizeOfTempMemory;
	if(memHit.Size() < requiredSize)
		memHit = std::move(DeviceMemory(requiredSize));

	// Populate pointers
	size_t offset = 0;
	std::uint8_t* dBasePtr = static_cast<uint8_t*>(memHit);
	dHits = reinterpret_cast<HitGMem*>(dBasePtr + offset);
	offset += sizeOfHits;
	dIds0 = reinterpret_cast<RayId*>(dBasePtr + offset);
	offset += sizeOfIds;
	dKeys0 = reinterpret_cast<HitKey*>(dBasePtr + offset);
	offset += sizeOfKeys;
	dIds1 = reinterpret_cast<RayId*>(dBasePtr + offset);
	offset += sizeOfIds;
	dKeys1 = reinterpret_cast<HitKey*>(dBasePtr + offset);
	offset += sizeOfKeys;
	dTempMemory = reinterpret_cast<void*>(dBasePtr + offset);
	offset += sizeOfTempMemory;
	assert(requiredSize == offset);

	dIds = dIds0;
	dKeys = dKeys0;

	// Initialize memory
	CudaSystem::GPUCallX(leaderDeviceId, 0, 0,
						 ResetHitIds,
						 dKeys, dIds, dHits,
						 static_cast<uint32_t>(rayCount));
}

void RayMemory::SortKeys(RayId*& ids, HitKey*& keys, 
						 size_t count,
						 const Vector2i& bitRange)
{
	// Sort Call over buffers
	cub::DoubleBuffer<HitKey> dbKeys(dKeys, (dKeys == dKeys0) ? dKeys1 : dKeys0);
	cub::DoubleBuffer<RayId> dbIds(dIds, (dIds == dIds0) ? dIds1 : dIds0);
	cub::DeviceRadixSort::SortPairs(dTempMemory, tempMemorySize,
									dbKeys, dbIds,
									static_cast<int>(count),
									bitRange[0], bitRange[1]);
	CUDA_KERNEL_CHECK();

	ids = dbIds.Current();
	keys = dbKeys.Current();
	dIds = ids;
	dKeys = keys;
}

RayPartitions<uint32_t> RayMemory::Partition(uint32_t& rayCount,
											 const Vector2i& bitRange)
{
	// Use double buffers for partition auxilary data		
	RayId* dEmptyIds = (dIds == dIds0) ? dIds1 : dIds0;
	HitKey* dEmptyKeys = (dKeys == dKeys0) ? dKeys1 : dKeys0;

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
	CudaSystem::GPUCallX(leaderDeviceId, 0, 0,
						 FindSplitsSparse,
						 dSparseSplitIndices, dKeys, rayCount, bitRange);
	// Make Splits Dense
	// From dEmptyKeys -> dEmptyIds	
	size_t selectTempMemorySize = (tempMemorySize - sizeof(uint32_t));
	CUDA_CHECK(cub::DeviceSelect::If(dSelectTempMemory, selectTempMemorySize,
									 dSparseSplitIndices, dDenseSplitIndices, dSelectCount,
									 static_cast<int>(rayCount),
									 ValidSplit()));

	// Copy Reduced Count
	uint32_t hSelectCount;
	CUDA_CHECK(cudaMemcpy(&hSelectCount, dSelectCount,
						  sizeof(uint32_t), cudaMemcpyDeviceToHost));


	// Find The Hit Keys for each split
	// From dEmptyIds, dKeys -> dEmptyKeys
	HitKey* dDenseKeys = reinterpret_cast<HitKey*>(dSparseSplitIndices);
	CudaSystem::GPUCallX(leaderDeviceId, 0, 0,
						 FindSplitKeys,
						 dDenseKeys,
						 dDenseSplitIndices,
						 dKeys,
						 hSelectCount,
						 bitRange);

	// We need to get dDenseIndices & dDenseKeys
	// Memcopy to vectors
	std::vector<HitKey> hDenseKeys(hSelectCount);
	std::vector<uint32_t> hDenseIndices(hSelectCount + 1);
	CUDA_CHECK(cudaMemcpy(hDenseKeys.data(), dDenseKeys,
						  sizeof(HitKey) * hSelectCount,
						  cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(hDenseIndices.data(), dDenseSplitIndices,
						  sizeof(uint32_t) * hSelectCount,
						  cudaMemcpyDeviceToHost));

	// Save old ray count
	uint32_t initialRayCount = rayCount;

	// If last split contains empty / invalids
	// Do not add it to partition list
	uint32_t InvalidKeyPartition = InvalidKey;
	InvalidKeyPartition >>= bitRange[0];
	InvalidKeyPartition &= ((0x1 << (bitRange[1] - bitRange[0])) - 1);
	if(hDenseKeys.back() == InvalidKeyPartition)
	{
		// Last one contains invalid rays
		// Do not add rayCount
		rayCount = hDenseIndices.back();
		hSelectCount--;
	}

	// Construct The Set
	// Add extra index to end as rayCount for cleaner code
	hDenseIndices.back() = initialRayCount;
	RayPartitions<uint32_t> partitions;
	for(uint32_t i = 0; i < hSelectCount; i++)
	{
		uint32_t id = hDenseKeys[i];
		uint32_t offset = hDenseIndices[i];
		size_t count = hDenseIndices[i + 1] - hDenseIndices[i];
		partitions.emplace(ArrayPortion<uint32_t>{id,offset, count});
	}
	// Done!
	return partitions;
}

void RayMemory::FillRayIdsForSort(uint32_t rayCount)
{
	CudaSystem::GPUCallX(leaderDeviceId, 0, 0,
						 FillHitIdsForSort,
						 dKeys, dIds, dHits,
						 static_cast<uint32_t>(rayCount));
}