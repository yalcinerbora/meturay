#include "RayMemory.h"
#include "RayLib/CudaConstants.h"
#include <cub/cub.cuh>
#include <cstddef>

__global__ void FillHitIds(HitId* gIds, uint32_t rayCount)
{
	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; 
		globalId += blockDim.x * gridDim.x)
	{
		gIds[globalId].rayId = globalId;
		gIds[globalId].innerId = 0xFFFFFFFF;
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
	ResizeRayMemory(dRayIn, dRayAuxIn, memIn, rayCount, perRayAuxSize);
}

void RayMemory::ResizeRayOut(size_t rayCount, size_t perRayAuxSize)
{
	ResizeRayMemory(dRayOut, dRayAuxOut, memOut, rayCount, perRayAuxSize);
}

void RayMemory::SwapRays(size_t rayCount)
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

	size_t sizeOfIds = sizeof(HitId) * rayCount;
	sizeOfIds = AlignByteCount * ((sizeOfIds + (AlignByteCount - 1)) / AlignByteCount);

	// Find out sort auxiliary storage
	cub::DoubleBuffer<HitKey> dbKeys;
	cub::DoubleBuffer<HitId> dbIds;
	size_t sizeOfSort = 0;
	cub::DeviceRadixSort::SortPairs(nullptr, sizeOfSort,
									dbKeys, dbIds,
									static_cast<int>(rayCount));

	size_t requiredSize = (sizeOfIds + sizeOfKeys) * 2 + sizeOfSort;
	if(memHit.Size() < requiredSize)
		memHit = std::move(DeviceMemory(requiredSize));

	// Populate pointers
	size_t offset = 0;
	std::uint8_t* dHit = static_cast<uint8_t*>(memHit);
	dIds0 = reinterpret_cast<HitId*>(dHit + offset);
	offset += sizeOfIds;
	dIds1 = reinterpret_cast<HitId*>(dHit + offset);
	offset += sizeOfIds;
	dKeys0 = reinterpret_cast<HitKey*>(dHit + offset);
	offset += sizeOfKeys;
	dKeys1 = reinterpret_cast<HitKey*>(dHit + offset);
	offset += sizeOfKeys;
	dSortAuxiliary = reinterpret_cast<void*>(dHit + offset);
	offset += sizeOfSort;
	assert(requiredSize == offset);

	dIds = dIds0;
	dKeys = dKeys0;

	// Initialize memory
	CUDA_CHECK(cudaMemset(dKeys, 0xFFFFFFFF, rayCount * sizeof(HitKey)));
	CudaSystem::GPUCallX(0, 0, 0, FillHitIds, dIds, static_cast<uint32_t>(rayCount));
}

void RayMemory::SortKeys(HitId*& ids, HitKey*& keys, 
						 size_t count,
						 const Vector2i& bitRange)
{
	// Sort Call over buffers
	size_t sizeOfSort = 0;
	cub::DoubleBuffer<HitKey> dbKeys(dKeys, (dKeys == dKeys0) ? dKeys1 : dKeys0);
	cub::DoubleBuffer<HitId> dbIds(dIds, (dIds == dIds0) ? dIds1 : dIds0);	
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

//template<class T>
//RayRecordGMem RayHitMemory<T>::RayStackIn()
//{
//	return rayStackIn;
//}
//
//RayRecordGMem RayHitMemory<T>::RayStackOut()
//{
//	return rayStackOut;
//}
//
//HitRecordGMem RayHitMemory<T>::HitRecord()
//{
//	return hitRecord;
//}
//
//const ConstRayRecordGMem RayHitMemory<T>::RayStackIn() const
//{
//	return rayStackIn;
//}
//
//const ConstRayRecordGMem RayHitMemory<T>::RayStackOut() const
//{
//	return rayStackOut;
//}
//
//const ConstHitRecordGMem RayHitMemory<T>::HitRecord() const
//{
//	return hitRecord;
//}

//void RayHitMemory<T>::AllocForCameraRays(size_t rayCount)
//{
//	memRayIn = std::move(DeviceMemory(TotalMemoryForRay(rayCount)));
//	//memHit = std::move(DeviceMemory(TotalMemoryForHit(rayCount)));
//	rayStackIn = GenerateRayPtrs(memRayIn, rayCount);
//	//hitRecord = GenerateHitPtrs(memHit, rayCount);
//}

//MaterialRays RayHitMemory<T>::ResizeRayIn(const MaterialRays& current,
//									const MaterialRays& external,
//									const MaterialRaysCPU& rays,
//									const MaterialHitsCPU& hits)
//{
//	// Determine total counts
//	size_t offset = 0;
//	MaterialRays mergedRays;
//	for(const auto& portion : current)
//	{
//		size_t portionCount = portion.count;
//
//		// Find external if avail
//		ArrayPortion<uint32_t> key = {portion.portionId};		
//		const auto loc = external.find(key);
//		if(loc != external.end())
//		{
//			portionCount += loc->count;
//		}
//
//		key.offset = offset;
//		key.count = portionCount;
//		mergedRays.emplace(key);
//		offset += portionCount;
//	}
//
//	// Generate new memory
//	DeviceMemory newRayMem = DeviceMemory(TotalMemoryForRay(offset));
//	DeviceMemory newHitMem = DeviceMemory(TotalMemoryForHit(offset));
//	RayRecordGMem newRays = GenerateRayPtrs(newRayMem, offset);
//	HitRecordGMem newHits = GenerateHitPtrs(newHitMem, offset);
//
//	// Copy from old memory
//	for(const auto& portion : mergedRays)
//	{
//		size_t copyOffset = 0;
//		RayRecordGMem newRayPortion = newRays;
//		newRayPortion.dirAndPixId += portion.offset;
//		newRayPortion.posAndMedium += portion.offset;
//		newRayPortion.radAndSampId += portion.offset;
//
//		HitRecordGMem newHitPortion = newHits;
//		newHitPortion.baryAndObjId += portion.offset;
//		newHitPortion.triId += portion.offset;
//
//		// Current if avail
//		ArrayPortion<uint32_t> key = {portion.portionId};
//		const auto loc = current.find(key);
//		if(loc != current.end())
//		{
//			copyOffset += loc->count;
//
//			// Ray
//			RayRecordGMem oldRayPortion = rayStackIn;
//			oldRayPortion.dirAndPixId += loc->offset;
//			oldRayPortion.posAndMedium += loc->offset;
//			oldRayPortion.radAndSampId += loc->offset;
//			CUDA_CHECK(cudaMemcpy(newRayPortion.dirAndPixId,
//								  oldRayPortion.dirAndPixId,
//								  sizeof(Vec3AndUInt) * loc->count,
//								  cudaMemcpyDeviceToDevice));
//			CUDA_CHECK(cudaMemcpy(newRayPortion.posAndMedium,
//								  oldRayPortion.posAndMedium,
//								  sizeof(Vector4) * loc->count,
//								  cudaMemcpyDeviceToDevice));
//			CUDA_CHECK(cudaMemcpy(newRayPortion.radAndSampId,
//								  oldRayPortion.radAndSampId,
//								  sizeof(Vec3AndUInt) * loc->count,
//								  cudaMemcpyDeviceToDevice));
//			// Hit
//			HitRecordGMem oldHitPortion = hitRecord;
//			oldHitPortion.baryAndObjId += loc->offset;
//			oldHitPortion.triId += loc->offset;	
//			CUDA_CHECK(cudaMemcpy(newHitPortion.baryAndObjId,
//								  oldHitPortion.baryAndObjId,
//								  sizeof(Vec3AndUInt) * loc->count,
//								  cudaMemcpyDeviceToDevice));
//			CUDA_CHECK(cudaMemcpy(newHitPortion.triId,
//								  oldHitPortion.triId,
//								  sizeof(unsigned int) * loc->count,
//								  cudaMemcpyDeviceToDevice));
//		}
//
//		// Check if cpu has some data
//		const auto locCPU = external.find(key);
//		if(locCPU != external.end())
//		{			
//			// Ray
//			const RayRecordCPU& rayCPU = rays.at(key.portionId);
//			CUDA_CHECK(cudaMemcpy(newRayPortion.dirAndPixId + copyOffset,
//								  rayCPU.dirAndPixId.data(),
//								  sizeof(Vec3AndUInt) * locCPU->count,
//								  cudaMemcpyHostToDevice));
//			CUDA_CHECK(cudaMemcpy(newRayPortion.posAndMedium + copyOffset,
//								  rayCPU.posAndMedium.data(),
//								  sizeof(Vector4) * locCPU->count,
//								  cudaMemcpyHostToDevice));
//			CUDA_CHECK(cudaMemcpy(newRayPortion.radAndSampId + copyOffset,
//								  rayCPU.radAndSampId.data(),
//								  sizeof(Vec3AndUInt) * locCPU->count,
//								  cudaMemcpyHostToDevice));
//			// Hit
//			const HitRecordCPU& hitCPU = hits.at(key.portionId);
//			CUDA_CHECK(cudaMemcpy(newHitPortion.baryAndObjId + copyOffset,
//								  hitCPU.baryAndObjId.data(),
//								  sizeof(Vec3AndUInt) * locCPU->count,
//								  cudaMemcpyHostToDevice));
//			CUDA_CHECK(cudaMemcpy(newHitPortion.triId + copyOffset,
//								  hitCPU.triId.data(),
//								  sizeof(unsigned int) * locCPU->count,
//								  cudaMemcpyHostToDevice));
//		}
//	}
//
//	// Change Memory
//	rayStackIn = newRays;
//	hitRecord = newHits;
//	memRayIn = std::move(newRayMem);
//	memHit = std::move(newHitMem);
//	return mergedRays;
//}
//
//void RayHitMemory<T>::ResizeRayOut(size_t rayCount)
//{
//	memRayOut = std::move(DeviceMemory(TotalMemoryForRay(rayCount)));
//	GenerateRayPtrs(memRayOut, rayCount);
//}
//
//void RayHitMemory<T>::SwapRays(size_t rayCount)
//{
//	// Get Ready For Hit loop
//	memRayIn = std::move(memRayOut);
//	memHit = std::move(DeviceMemory(TotalMemoryForHit(rayCount)));
//	// Pointers
//	rayStackIn = rayStackOut;
//	GenerateHitPtrs(memHit, rayCount);
//}