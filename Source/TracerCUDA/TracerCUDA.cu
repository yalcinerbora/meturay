#include <random>

#include "TracerCUDA.h"
#include "CameraKernels.cuh"
#include "RayLib/Camera.h"
#include "RayLib/CudaConstants.h"
#include "RayLib/Random.cuh"

#include "RayLib/Log.h"

RNGMemory::RNGMemory(uint32_t seed)
{
	assert(CudaSystem::GPUList().size() > 0);

	// CPU Mersenne Twister
	std::mt19937 rng;
	rng.seed(seed);

	// Determine GPU
	size_t totalCount = 0;
	for(const auto& gpu : CudaSystem::GPUList())
	{
		totalCount += gpu.RecommendedBlockCount() * StaticThreadPerBlock1D;
	}

	// Actual Allocation
	size_t totalSize = totalCount * sizeof(uint32_t);
	memRandom = std::move(DeviceMemory(totalSize));
	uint32_t* d_ptr = static_cast<uint32_t*>(memRandom);

	size_t totalOffset = 0;
	for(const auto& gpu : CudaSystem::GPUList())
	{
		randomStacks.emplace_back(RandomStackGMem{d_ptr + totalOffset});
		totalOffset += gpu.RecommendedBlockCount() * StaticThreadPerBlock1D;
	}
	assert(totalCount == totalOffset);

	// Init all seeds
	std::vector<uint32_t> seeds(totalCount);
	for(size_t i = 0; i < totalCount; i++)
	{
		d_ptr[i] = rng();
	}
}

RandomStackGMem RNGMemory::RandomStack(uint32_t gpuId)
{
	return randomStacks[gpuId];
}

size_t RNGMemory::SharedMemorySize(uint32_t gpuId)
{
	return StaticThreadPerBlock1D * sizeof(uint32_t);
}

RayRecordGMem RayMemory::GenerateRayPtrs(void* mem, size_t rayCount)
{
	RayRecordGMem record;

	// Ray Size
	size_t offset = 0;
	size_t totalAlloc = TotalMemoryForRay(rayCount);
	
	// Pointer determination
	byte* d_ptr = static_cast<byte*>(mem);	
	record.posAndMedium = reinterpret_cast<Vector4*>(d_ptr + offset);
	offset += rayCount * sizeof(Vector4);
	record.dirAndPixId = reinterpret_cast<Vec3AndUInt*>(d_ptr + offset);
	offset += rayCount * sizeof(Vec3AndUInt);
	record.radAndSampId = reinterpret_cast<Vec3AndUInt*>(d_ptr + offset);
	offset += rayCount * sizeof(Vec3AndUInt);

	// Check that we did properly
	assert(offset == totalAlloc);
	return record;
}

HitRecordGMem RayMemory::GenerateHitPtrs(void* mem, size_t rayCount)
{
	HitRecordGMem record;

	// Hit Size
	size_t offset = 0;
	size_t totalAlloc = TotalMemoryForHit(rayCount);

	// Pointer determination
	byte* d_ptr = static_cast<byte*>(mem);
	record.baryAndObjId = reinterpret_cast<Vec3AndUInt*>(d_ptr + offset);
	offset += rayCount * sizeof(Vec3AndUInt);
	record.triId = reinterpret_cast<unsigned int*>(d_ptr + offset);
	offset += rayCount * sizeof(unsigned int);

	// Check that we did properly
	assert(offset == totalAlloc);
	return record;
}

size_t RayMemory::TotalMemoryForRay(size_t rayCount)
{
	static_assert(sizeof(RayRecord) == sizeof(Vector4) + sizeof(Vec3AndUInt) * 2,
				  "Ray record size sanity check.");
	return sizeof(RayRecord) * rayCount;
}

size_t RayMemory::TotalMemoryForHit(size_t rayCount)
{
	static_assert(sizeof(::HitRecord) == sizeof(Vector3) + sizeof(unsigned int) * 2 + sizeof(float),
				  "Hit record size sanity check.");
	return sizeof(::HitRecord) * rayCount;
}

RayMemory::RayMemory()
	: rayStackIn{nullptr, nullptr, nullptr}
	, rayStackOut{nullptr, nullptr, nullptr}
	, hitRecord{nullptr, nullptr}
{}

RayRecordGMem RayMemory::RayStackIn()
{
	return rayStackIn;
}

RayRecordGMem RayMemory::RayStackOut()
{
	return rayStackOut;
}

HitRecordGMem RayMemory::HitRecord()
{
	return hitRecord;
}

const ConstRayRecordGMem RayMemory::RayStackIn() const
{
	return rayStackIn;
}

const ConstRayRecordGMem RayMemory::RayStackOut() const
{
	return rayStackOut;
}

const ConstHitRecordGMem RayMemory::HitRecord() const
{
	return hitRecord;
}

void RayMemory::AllocForCameraRays(size_t rayCount)
{
	memRayIn = std::move(DeviceMemory(TotalMemoryForRay(rayCount)));
	memHit = std::move(DeviceMemory(TotalMemoryForHit(rayCount)));
}

MaterialRays RayMemory::ResizeRayIn(const MaterialRays& current,
									const MaterialRays& external,
									const MaterialRaysCPU& rays,
									const MaterialHitsCPU& hits)
{
	// Determine total counts
	size_t offset = 0;
	MaterialRays mergedRays;
	for(const auto& portion : current)
	{
		size_t portionCount = portion.count;

		// Find external if avail
		ArrayPortion<uint32_t> key = {portion.portionId};		
		const auto loc = external.find(key);
		if(loc != external.end())
		{
			portionCount += loc->count;
		}

		key.offset = offset;
		key.count = portionCount;
		mergedRays.emplace(key);
		offset += portionCount;
	}

	// Generate new memory
	DeviceMemory newRayMem = DeviceMemory(TotalMemoryForRay(offset));
	DeviceMemory newHitMem = DeviceMemory(TotalMemoryForHit(offset));
	RayRecordGMem newRays = GenerateRayPtrs(newRayMem, offset);
	HitRecordGMem newHits = GenerateHitPtrs(newHitMem, offset);

	// Copy from old memory
	for(const auto& portion : mergedRays)
	{
		size_t copyOffset = 0;
		RayRecordGMem newRayPortion = newRays;
		newRayPortion.dirAndPixId += portion.offset;
		newRayPortion.posAndMedium += portion.offset;
		newRayPortion.radAndSampId += portion.offset;

		HitRecordGMem newHitPortion = newHits;
		newHitPortion.baryAndObjId += portion.offset;
		newHitPortion.triId += portion.offset;

		// Current if avail
		ArrayPortion<uint32_t> key = {portion.portionId};
		const auto loc = current.find(key);
		if(loc != current.end())
		{
			copyOffset += loc->count;

			// Ray
			RayRecordGMem oldRayPortion = rayStackIn;
			oldRayPortion.dirAndPixId += loc->offset;
			oldRayPortion.posAndMedium += loc->offset;
			oldRayPortion.radAndSampId += loc->offset;
			CUDA_CHECK(cudaMemcpy(newRayPortion.dirAndPixId,
								  oldRayPortion.dirAndPixId,
								  sizeof(Vec3AndUInt) * loc->count,
								  cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(newRayPortion.posAndMedium,
								  oldRayPortion.posAndMedium,
								  sizeof(Vector4) * loc->count,
								  cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(newRayPortion.radAndSampId,
								  oldRayPortion.radAndSampId,
								  sizeof(Vec3AndUInt) * loc->count,
								  cudaMemcpyDeviceToDevice));
			// Hit
			HitRecordGMem oldHitPortion = hitRecord;
			oldHitPortion.baryAndObjId += loc->offset;
			oldHitPortion.triId += loc->offset;	
			CUDA_CHECK(cudaMemcpy(newHitPortion.baryAndObjId,
								  oldHitPortion.baryAndObjId,
								  sizeof(Vec3AndUInt) * loc->count,
								  cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(newHitPortion.triId,
								  oldHitPortion.triId,
								  sizeof(unsigned int) * loc->count,
								  cudaMemcpyDeviceToDevice));
		}

		// Check if cpu has some data
		const auto locCPU = external.find(key);
		if(locCPU != external.end())
		{			
			// Ray
			const RayRecordCPU& rayCPU = rays.at(key.portionId);
			CUDA_CHECK(cudaMemcpy(newRayPortion.dirAndPixId + copyOffset,
								  rayCPU.dirAndPixId.data(),
								  sizeof(Vec3AndUInt) * locCPU->count,
								  cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(newRayPortion.posAndMedium + copyOffset,
								  rayCPU.posAndMedium.data(),
								  sizeof(Vector4) * locCPU->count,
								  cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(newRayPortion.radAndSampId + copyOffset,
								  rayCPU.radAndSampId.data(),
								  sizeof(Vec3AndUInt) * locCPU->count,
								  cudaMemcpyHostToDevice));
			// Hit
			const HitRecordCPU& hitCPU = hits.at(key.portionId);
			CUDA_CHECK(cudaMemcpy(newHitPortion.baryAndObjId + copyOffset,
								  hitCPU.baryAndObjId.data(),
								  sizeof(Vec3AndUInt) * locCPU->count,
								  cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(newHitPortion.triId + copyOffset,
								  hitCPU.triId.data(),
								  sizeof(unsigned int) * locCPU->count,
								  cudaMemcpyHostToDevice));
		}
	}

	// Change Memory
	rayStackIn = newRays;
	hitRecord = newHits;
	memRayIn = std::move(newRayMem);
	memHit = std::move(newHitMem);
	return mergedRays;
}

void RayMemory::ResizeRayOut(size_t rayCount)
{
	memRayOut = std::move(DeviceMemory(TotalMemoryForRay(rayCount)));
}

void RayMemory::SwapRays(size_t rayCount)
{
	// Get Ready For Hit loop
	memRayIn = std::move(memRayOut);
	memHit = std::move(DeviceMemory(TotalMemoryForHit(rayCount)));
	// Pointers
	rayStackIn = rayStackOut;
	GenerateHitPtrs(memHit, rayCount);
}

void TracerCUDA::SetScene(const std::string& sceneFileName)
{
	scene = std::move(SceneGPU(sceneFileName));
}

void TracerCUDA::SetParams(const TracerParameters& p)
{
	parameters = p;
}

void TracerCUDA::GenerateSceneAccelerator()
{
	// TODO:
}

void TracerCUDA::GenerateAccelerator(uint32_t objId)
{
	// TODO:
}

void TracerCUDA::ReportionImage(const Vector2ui& offset,
								const Vector2ui& size)
{
	imageOffset = offset;
	imageSegmentSize = size;
}

void TracerCUDA::ResizeImage(const Vector2ui& resolution)
{
	imageResolution = resolution;
}

void TracerCUDA::ResetImage()
{
	size_t pixelCount = imageSegmentSize[0] * imageSegmentSize[1];
	CUDA_CHECK(cudaMemset(outputImage.imagePtr, 0x0, sizeof(Vector3f) * pixelCount))
}

std::vector<Vector3f> TracerCUDA::GetImage()
{
	CUDA_CHECK(cudaDeviceSynchronize());
	size_t pixelCount = imageSegmentSize[0] * imageSegmentSize[1];
	std::vector<Vector3> out(pixelCount);
	std::memcpy(out.data(), outputImage.imagePtr, sizeof(Vector3f) * pixelCount);

	return out;
}

void TracerCUDA::AssignAllMaterials()
{

}

void TracerCUDA::AssignMaterial(uint32_t matId)
{

}

void TracerCUDA::LoadMaterial(uint32_t matId)
{

}

void TracerCUDA::UnloadMaterial(uint32_t matId)
{

}

void TracerCUDA::GenerateCameraRays(const CameraPerspective& camera,
									const uint32_t samplePerPixel)
{
	// Single GPU Camera Ray Generation
	const uint32_t gpuId = 0;

	const uint32_t blockCount = CudaSystem::GPUList()[gpuId].RecommendedBlockCount();
	const uint32_t threadCount = StaticThreadPerBlock1D;
	const size_t sharedSize = rngMemory.SharedMemorySize(gpuId);

	KCGenerateCameraRays<<<blockCount, threadCount, sharedSize>>>(rayMemory.RayStackIn(),
																  rngMemory.RandomStack(gpuId),
																  camera,
																  samplePerPixel,
																  imageResolution,
																  imageOffset,
																  imageSegmentSize);
	CUDA_KERNEL_CHECK();

	//// DEBUG
	//// Check Rays
	//CUDA_CHECK(cudaDeviceSynchronize());

	//for(int i = 0; i < totalRayCount; i++)
	//{
	//	Vector3 dir = rayStackIn.dirAndPixId[i].vec;
	//	Vector3 pos = Vector3(rayStackIn.posAndMedium[i][0], 
	//						  rayStackIn.posAndMedium[i][1], 
	//						  rayStackIn.posAndMedium[i][2]);

	//	METU_LOG("d(%f, %f, %f) p(%f, %f, %f)",
	//			 dir[0], dir[1], dir[2],
	//			 pos[0], pos[1], pos[2]);
	//}
}

void TracerCUDA::HitRays()
{

}

void TracerCUDA::GetMaterialRays(RayRecordCPU&, HitRecordCPU&,
								 uint32_t rayCount, uint32_t matId)
{

}

void TracerCUDA::AddMaterialRays(const RayRecordCPU&, const HitRecordCPU&,
								 uint32_t rayCount, uint32_t matId)
{

}

void TracerCUDA::BounceRays()
{
	// TODO::

	// Rays are sorted and ready for material calls
	// For each loaded material launch

}

uint32_t TracerCUDA::RayCount()
{
	return 0;
}

TracerCUDA::TracerCUDA()
	: currentRayCount(0)
	, parameters{}
	, imageSegmentSize(0, 0)
	, imageOffset(0, 0)
	, imageResolution(0, 0)
{}

TracerCUDA::~TracerCUDA()
{

}

void TracerCUDA::Initialize()
{
	// Generate CUDA Tracer Interface
	if(!CudaSystem::Initialize())
	{
		METU_ERROR_LOG("Unable to Init CUDA");
	}

	// Cuda System Initalized

}

//// DELETE THOSE
//void TracerCUDA::LoadBackgroundCubeMap(const std::vector<float>& cubemap)
//{
//
//}
//
//void TracerCUDA::LoadFluidToGPU(const std::vector<float>& velocityDensity,
//								const Vector3ui& size)
//{
//	// Copy Density
//	velocityDensityTexture.Copy(reinterpret_cast<const byte*>(velocityDensity.data()), size);
//}
//
//void TracerCUDA::CS568GenerateCameraRays(const CameraPerspective& cam,
//										 const Vector2ui resolution,
//										 const uint32_t samplePerPixel)
//{
//	totalRayCount = resolution[0] * resolution[1] * samplePerPixel * samplePerPixel;
//
//	AllocateRayStack(resolution[0] * resolution[1] * samplePerPixel * samplePerPixel);
//	AllocateRandomStack();
//	AllocateImage(resolution);
//
//	uint32_t blockCount = CudaSystem::GPUList()[0].RecommendedBlockCount();
//	uint32_t threadCount = StaticThreadPerBlock1D;
//	uint32_t sharedSize = sizeof(uint32_t) * StaticThreadPerBlock1D;
//
//	KCGenerateCameraRays<<<blockCount, threadCount, sharedSize>>>(rayStackIn,
//																  random,
//																  cam,
//																  samplePerPixel,
//																  resolution,
//																  Vector2ui(0, 0),
//																  resolution);
//	CUDA_KERNEL_CHECK();
//
//	//// DEBUG
//	//// Check Rays
//	//CUDA_CHECK(cudaDeviceSynchronize());
//
//	//for(int i = 0; i < totalRayCount; i++)
//	//{
//	//	Vector3 dir = rayStackIn.dirAndPixId[i].vec;
//	//	Vector3 pos = Vector3(rayStackIn.posAndMedium[i][0], 
//	//						  rayStackIn.posAndMedium[i][1], 
//	//						  rayStackIn.posAndMedium[i][2]);
//
//	//	METU_LOG("d(%f, %f, %f) p(%f, %f, %f)",
//	//			 dir[0], dir[1], dir[2],
//	//			 pos[0], pos[1], pos[2]);
//	//}
//}


//__device__ bool SnapToInside(float& distance,
//
//							 const RayF& ray,
//							 const Vector3f liquidBottomLeft,
//							 const Vector3f liquidWorldLength)
//{
//	Vector3f pos;
//	distance = ray.IntersectsAABB(pos,
//								  liquidBottomLeft,
//								  liquidBottomLeft + liquidWorldLength);
//	
//	if(distance == FLT_MAX)
//		return false;
//	return true;
//}
//
//__device__ bool CheckIntersection(float& distance,
//								  float& density,
//								  Vector3f& velocity,
//								  
//								  const RayF& ray,
//
//								  const Texture3<float4>& liquidTexture,
//								  const Vector3ui& liquidDim,
//								  const Vector3f& liquidBottomLeft,
//								  const Vector3f& liquidWorldLength)
//{
//	Vector3 gridSpan = liquidWorldLength / Vector3f(liquidDim[0],
//													liquidDim[1],
//													liquidDim[2]);
//	Vector3 gridOffset = (((ray.getPosition() - liquidBottomLeft) / gridSpan).FloorSelf()) * gridSpan;
//	Vector3 gridWorld = liquidBottomLeft + gridOffset;
//	
//	// Calculate next distance
//	distance = IntersectDistance(ray,
//								 gridWorld,
//								 gridWorld + gridSpan);
//
//	// Sample Texture
//	Vector3 normalizedCoords = (ray.getPosition() - liquidBottomLeft) / Vector3f(liquidDim[0],
//																				 liquidDim[1],
//																				 liquidDim[2]);
//	float4 data = liquidTexture(normalizedCoords);
//
//	// Push Texture
//	density = data.w;
//	velocity[0] = data.x;
//	velocity[1] = data.y;
//	velocity[2] = data.z;
//
//	// FLT max means no intersection
//	if(distance == FLT_MAX)
//		return false;
//	return true;
//}

//// Delete this kernel
//__global__ void KernelLoop(Vector3f* outImage,
//
//						   ConstRayRecordGMem gRays,
//						   RandomStackGMem gRand,
//						   const uint32_t totalRayCount,
//
//						   const Vector3f backgroundColor,
//
//						   // Texture Related
//						   Texture3<float4> liquidTexture,
//						   const Vector3ui liquidDim,
//						   const Vector3f liquidBottomLeft,
//						   const Vector3f liquidWorldLength)
//{
//	extern __shared__ uint32_t sRandState[];
//	RandomGPU rng(gRand.state, sRandState);
//
//	const uint32_t totalWorkCount = totalRayCount;
//
//	// Kernel Grid-Stride Loop
//	for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
//		threadId < totalWorkCount;
//		threadId += (blockDim.x * gridDim.x))
//	{
//		// Load Ray to registers
//		RayRecord rayData(gRays, threadId);
//		rayData.totalRadiance = Vector3(1.0f);
//
//		// Accumulate
//		Vector3f totalIllumination = Zero3;
//		float totalOcclusion = 0.0f;
//		
//		// Loop until ray hits
//		RayF r = rayData.ray;
//		
//		Vector3 waterColor = Vector3(0.0f, 0.0f, 1.0f);
//
//		// Snap to water grid
//		float distance = 0.0f;
//		if(SnapToInside(distance, r, liquidBottomLeft, liquidWorldLength))
//		{
//			//if(threadId == (512 * 256 + 256))
//			//	printf("(%f, %f, %f)\n",
//			//		   r.getPosition()[0],
//			//		   r.getPosition()[1],
//			//		   r.getPosition()[2]);
//
//			r.AdvanceSelf(distance + 0.001f);
//
//
//			Vector3 liquidTopRight = liquidBottomLeft + liquidWorldLength;
//
//			if(threadId == (512 * 256 + 256))
//			{
//				//printf("(%f, %f, %f)\n",
//				//	   r.getPosition()[0],
//				//	   r.getPosition()[1],
//				//	   r.getPosition()[2]);
//
//				//printf("(%f, %f, %f)\n",
//				//	  liquidBottomLeft[0],
//				//	  liquidBottomLeft[1],
//				//	  liquidBottomLeft[2]);
//
//				//printf("(%f, %f, %f)\n",
//				//	   liquidTopRight[0],
//				//	   liquidTopRight[1],
//				//	   liquidTopRight[2]);
//			}
//
//			while(r.getPosition()[0] > liquidBottomLeft[0] && 
//					r.getPosition()[0] < liquidTopRight[0] &&
//
//					r.getPosition()[1] > liquidBottomLeft[1] &&
//					r.getPosition()[1] < liquidTopRight[1] &&
//
//					r.getPosition()[2] > liquidBottomLeft[2] &&
//					r.getPosition()[2] < liquidTopRight[2])
//			{
//
//				//if(threadId == (512 * 256 + 256))
//				//   printf("looping\n");
//
//				// Sample Texture
//				Vector3 normalizedCoords = (r.getPosition() - liquidBottomLeft) / Vector3f(liquidWorldLength[0],
//																						   liquidWorldLength[1],
//																						   liquidWorldLength[2]);
//				float4 data = liquidTexture(normalizedCoords);
//
//				//if(threadId == (512 * 256 + 256))
//				//{
//				//	printf("(%f, %f, %f)\n",
//				//		   r.getPosition()[0],
//				//		   r.getPosition()[1],
//				//		   r.getPosition()[2]);
//				//	printf("(%f, %f, %f)\n",
//				//		   normalizedCoords[0],
//				//		   normalizedCoords[1],
//				//		   normalizedCoords[2]);
//				//	printf("%f\n", data.w);
//				//}
//
//				float density = fminf(data.w, 1.0f);
//				Vector3f velocity;
//				velocity[0] = data.x;
//				velocity[1] = data.y;
//				velocity[2] = data.z;
//
//				// Ray hits!
//				// Do stuff
//				totalIllumination += density * waterColor * (1.0f - totalOcclusion);
//				totalOcclusion += density * (1.0f - totalOcclusion);
//
//				// Advance ray
//				Vector3 gridSpan = liquidWorldLength / Vector3f(liquidDim[0],
//																liquidDim[1],
//																liquidDim[2]);
//				float spanAvg = gridSpan[0] + gridSpan[1] + gridSpan[2] * 0.33f;
//				//r.AdvanceSelf(spanAvg * 1.732);
//				r.AdvanceSelf(0.05f);
//			}
//
//		}
//
//		totalIllumination += backgroundColor * (1.0f - totalOcclusion);
//
//		// Write total illumination
//		float* pixelAdress = reinterpret_cast<float*>(outImage + rayData.pixelId);
//		atomicAdd(pixelAdress + 0, totalIllumination[0]);
//		atomicAdd(pixelAdress + 1, totalIllumination[1]);
//		atomicAdd(pixelAdress + 2, totalIllumination[2]);		
//	}
//}
//
//void TracerCUDA::LaunchRays(const Vector3f& backgroundColor,
//							const Vector3ui& textureSize,
//							const Vector3f& bottomLeft,
//							const Vector3f& length)
//{
//	uint32_t blockCount = CudaSystem::GPUList()[0].RecommendedBlockCount();
//	uint32_t threadCount = StaticThreadPerBlock1D;
//	uint32_t sharedSize = sizeof(uint32_t) * StaticThreadPerBlock1D;
//
//	KernelLoop<<<blockCount, threadCount, sharedSize>>>(dOutImage, rayStackIn, random,
//													    totalRayCount,
//													    backgroundColor,
//													    velocityDensityTexture,
//													    textureSize,
//													    bottomLeft,
//													    length);
//	CUDA_KERNEL_CHECK();
//}
//
//std::vector<Vector3> TracerCUDA::GetImage(const Vector2ui& resolution)
//{
//	CUDA_CHECK(cudaDeviceSynchronize());
//	size_t pixelCount = resolution[0] * resolution[1];
//	std::vector<Vector3> out(pixelCount);
//	memcpy(out.data(), dOutImage, sizeof(Vector3f) * pixelCount);
//		
//	return out;
//}
