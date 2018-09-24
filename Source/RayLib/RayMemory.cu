#include "RayMemory.h"
#include "RayLib/CudaConstants.h"
#include <cub/cub.cuh>
#include <cstddef>
#include <type_traits>

struct ValidSplit
{
	__device__ __host__ 
	__forceinline__ bool operator()(const uint32_t &a) const
	{
		return (a != RayMemory::InvalidData);
	}
};

__global__ void FillHitIds(HitKey* gKeys, RayId* gIds, uint32_t rayCount)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; 
		globalId += blockDim.x * gridDim.x)
	{
		gIds[globalId] = globalId;
		gKeys[globalId] = RayMemory::InvalidKey;
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
							  uint32_t* gDenseIds,
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

void RayMemory::Reset(size_t rayCount)
{}

void RayMemory::ResizeRayIn(size_t rayCount, size_t perRayAuxSize)
{
	memInMaxRayCount = rayCount;
	ResizeRayMemory(dRayIn, dRayAuxIn, memIn, rayCount, perRayAuxSize);
}

void RayMemory::ResizeRayOut(size_t rayCount, size_t perRayAuxSize)
{
	memOutMaxRayCount = rayCount;
	ResizeRayMemory(dRayOut, dRayAuxOut, memOut, rayCount, perRayAuxSize);
}

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

void RayMemory::ResizeHitMemory(size_t rayCount)
{
	// Align to proper memory strides
	size_t sizeOfKeys = sizeof(HitKey) * rayCount;
	sizeOfKeys = AlignByteCount * ((sizeOfKeys + (AlignByteCount - 1)) / AlignByteCount);

	size_t sizeOfIds = sizeof(RayId) * rayCount;
	sizeOfIds = AlignByteCount * ((sizeOfIds + (AlignByteCount - 1)) / AlignByteCount);

	// Find out sort auxiliary storage
	cub::DoubleBuffer<HitKey> dbKeys;
	cub::DoubleBuffer<RayId> dbIds;
	size_t sizeOfSort = 0;
	CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sizeOfSort,
											   dbKeys, dbIds,
											   static_cast<int>(rayCount)));
	

	// Check if while partitioning  double buffer data is 
	// enough for using (Unique and Scan) algos	
	size_t selectSize = 0;
	uint32_t* in = nullptr;
	uint32_t* out = nullptr;
	uint32_t* count = nullptr;
	CUDA_CHECK(cub::DeviceSelect::If(nullptr, selectSize,
									 in, out, count,
									 static_cast<int>(rayCount),
									 ValidSplit()));

	// Select algo reads from split locations and writes to backbuffer Ids (half is used)
	// uses backbuffer ids other half as auxiliary buffer
	// This code tries to increase it accordingly
	if(selectSize > (sizeOfIds / 2))
		sizeOfIds = 2 * (AlignByteCount * ((selectSize + (AlignByteCount - 1)) / AlignByteCount));

	// Finally allocate
	size_t requiredSize = (sizeOfIds + sizeOfKeys) * 2 + sizeOfSort;
	if(memHit.Size() < requiredSize)
		memHit = std::move(DeviceMemory(requiredSize));

	// Populate pointers
	size_t offset = 0;
	std::uint8_t* dHit = static_cast<uint8_t*>(memHit);
	dIds0 = reinterpret_cast<RayId*>(dHit + offset);
	offset += sizeOfIds;
	dKeys0 = reinterpret_cast<HitKey*>(dHit + offset);
	offset += sizeOfKeys;
	dIds1 = reinterpret_cast<RayId*>(dHit + offset);
	offset += sizeOfIds;
	dKeys1 = reinterpret_cast<HitKey*>(dHit + offset);
	offset += sizeOfKeys;
	dSortAuxiliary = reinterpret_cast<void*>(dHit + offset);
	offset += sizeOfSort;
	assert(requiredSize == offset);

	dIds = dIds0;
	dKeys = dKeys0;

	// Initialize memory
	CudaSystem::GPUCallX(leaderDeviceId, 0, 0, FillHitIds, dKeys, dIds, static_cast<uint32_t>(rayCount));
}

void RayMemory::SortKeys(RayId*& ids, HitKey*& keys, 
						 size_t count,
						 const Vector2i& bitRange)
{
	// Sort Call over buffers
	size_t sizeOfSort = 0;
	cub::DoubleBuffer<HitKey> dbKeys(dKeys, (dKeys == dKeys0) ? dKeys1 : dKeys0);
	cub::DoubleBuffer<RayId> dbIds(dIds, (dIds == dIds0) ? dIds1 : dIds0);
	cub::DeviceRadixSort::SortPairs(dSortAuxiliary, sizeOfSort,
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
	static_assert(sizeof(RayId) == 2 * sizeof(HitKey), "Size Mismatch");
	uint32_t* dSparseSplitIndices = dEmptyKeys;
	uint32_t* dDenseSplitIndices = reinterpret_cast<uint32_t*>(dEmptyIds);	
	uint32_t* dSelectCount = static_cast<uint32_t*>(dSortAuxiliary);
	void* dSelectAuxBuffer = dEmptyIds + memInMaxRayCount / 2;

	// Find Split Locations
	CudaSystem::GPUCallX(leaderDeviceId, 0, 0,
						 FindSplitsSparse,
						 dSparseSplitIndices, dKeys, rayCount, bitRange);
	// Make Splits Dense
	size_t auxMemCount = memInMaxRayCount * sizeof(uint32_t);
	CUDA_CHECK(cub::DeviceSelect::If(dSelectAuxBuffer, auxMemCount,
									 dSparseSplitIndices, dDenseSplitIndices, dSelectCount,
									 static_cast<int>(rayCount),
									 ValidSplit()));

	// Copy Reduced Count
	uint32_t hSelectCount;
	CUDA_CHECK(cudaMemcpy(&hSelectCount, dSelectCount,
						  sizeof(uint32_t), cudaMemcpyDeviceToHost));

	// Find The Hit Keys for each split
	// Change Sparse split indices since we used it
	HitKey* dDenseKeys = reinterpret_cast<HitKey*>(dSparseSplitIndices);
	CudaSystem::GPUCallX(leaderDeviceId, 0, 0,
						 FindSplitKeys,
						 dDenseKeys,
						 dDenseSplitIndices,
						 dKeys,
						 hSelectCount);

	// Memcopy Here
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