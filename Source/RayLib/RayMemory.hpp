#pragma once

//template <class T>
//RayGMem RayMemory<T>::GenerateRayPtrs(void* mem, size_t rayCount)
//{
//	RayGMem record;
//
//	// Ray Size
//	size_t offset = 0;
//	size_t totalAlloc = TotalMemoryForRay(rayCount);
//	
//	// Pointer determination
//	byte* d_ptr = static_cast<byte*>(mem);
//	recor
//
//	record.posAndMedium = reinterpret_cast<Vector4*>(d_ptr + offset);
//	offset += rayCount * sizeof(Vector4);
//	record.dirAndPixId = reinterpret_cast<Vec3AndUInt*>(d_ptr + offset);
//	offset += rayCount * sizeof(Vec3AndUInt);
//	record.radAndSampId = reinterpret_cast<Vec3AndUInt*>(d_ptr + offset);
//	offset += rayCount * sizeof(Vec3AndUInt);
//
//	// Check that we did properly
//	assert(offset == totalAlloc);
//	return record;
//}
//
//template <class T>
//HitGMem RayMemory<T>::GenerateHitPtrs(void* mem, size_t rayCount)
//{
//	HitGMem record;
//
//	// Hit Size
//	size_t offset = 0;
//	size_t totalAlloc = TotalMemoryForHit(rayCount);
//
//	// Pointer determination
//	byte* d_ptr = static_cast<byte*>(mem);
//	record.baryAndObjId = reinterpret_cast<Vec3AndUInt*>(d_ptr + offset);
//	offset += rayCount * sizeof(Vec3AndUInt);
//	record.triId = reinterpret_cast<unsigned int*>(d_ptr + offset);
//	offset += rayCount * sizeof(unsigned int);
//	record.distance = reinterpret_cast<float*>(d_ptr + offset);
//	offset += rayCount * sizeof(float);
//
//	// Check that we did properly
//	assert(offset == totalAlloc);
//	return record;
//}

template<class T>
RayMemory<T>::RayMemory()
{}

template<class T>
void RayMemory<T>::Reset(size_t rayCount)
{}

template<class T>
void RayMemory<T>::ResizeRayIn(size_t rayCount)
{}

template<class T>
void RayMemory<T>::ResizeRayOut(size_t rayCount)
{}

template<class T>
void RayMemory<T>::ResizeHit(size_t rayCount)
{}

template<class T>
void RayMemory<T>::SwapRays(size_t rayCount)
{}

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