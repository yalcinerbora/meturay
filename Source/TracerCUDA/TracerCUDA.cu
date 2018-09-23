#include <random>

#include "TracerCUDA.h"
#include "RayLib/Camera.h"
#include "RayLib/CudaConstants.h"
#include "RayLib/Log.h"
#include "RayLib/SceneIO.h"
#include "RayLib/Random.cuh"
#include "RayLib/ImageIO.h"

void TracerCUDA::SendError(TracerError e, bool isFatal)
{
	if(errorFunc) errorFunc(e);
	healthy = isFatal;
}

void TracerCUDA::HitRays()
{

}

void TracerCUDA::ShadeRays()
{

}

TracerCUDA::TracerCUDA()
	: rayDelegateFunc(nullptr)
	, errorFunc(nullptr)
	, analyticFunc(nullptr)
	, imageFunc(nullptr)
	, hitManager(DefaultHitmanOptions)
{}

void TracerCUDA::Initialize(uint32_t seed)
{
	// Device initalization
	CudaSystem::Initialize();

	// Select a leader device that is responsible
	// for sorting and partitioning works
	// for different materials / accelerators
	// TODO: Determine a leader Device
	rayMemory.SetLeaderDevice(0);	   
}

void TracerCUDA::SetTime(double seconds)
{}

void TracerCUDA::SetParams(const TracerParameters&)
{}

void TracerCUDA::SetScene(const std::string& sceneFileName)
{}

void TracerCUDA::GenerateSceneAccelerator()
{}

void TracerCUDA::GenerateAccelerator(uint32_t objId)
{}

void TracerCUDA::AssignAllMaterials()
{}

void TracerCUDA::AssignMaterial(uint32_t matId)
{}

void TracerCUDA::LoadMaterial(uint32_t matId)
{}

void TracerCUDA::UnloadMaterial(uint32_t matId)
{}

void TracerCUDA::GenerateCameraRays(const CameraPerspective& camera,
									const uint32_t samplePerPixel)
{
	currentRayCount = 512;
	rayMemory.ResizeRayIn(4096, 0);

	//size_t samples = samplePerPixel *  samplePerPixel;
	//Vector2i pixel2D = outputImage.
	//size_t rayCount = outputImage.
	//rayMemory.Reset(samples * )


}

bool TracerCUDA::Continue()
{
	return (currentRayCount > 0) && healthy;
}

void TracerCUDA::Render()
{
	if(!healthy) return;
	if(currentRayCount == 0) return;
	
	// We know that we have some valid rays in the system
	// First we hit rays until we could not find anything
	rayMemory.ResizeHitMemory(currentRayCount);
	hitManager.Process(rayMemory, currentRayCount);

	// Then we use these rays to shade and create more rays
}

void TracerCUDA::FinishSamples() 
{
	if(!healthy) return;
}

bool TracerCUDA::IsCrashed()
{
	return (!healthy);
}

void TracerCUDA::AddMaterialRays(const RayRecordCPU&, const HitRecordCPU&,
								 uint32_t rayCount, uint32_t matId)
{}

void TracerCUDA::SetImagePixelFormat(PixelFormat)
{}

void TracerCUDA::ReportionImage(const Vector2ui& offset,
								const Vector2ui& size)
{}

void TracerCUDA::ResizeImage(const Vector2ui& resolution)
{}

void TracerCUDA::ResetImage()
{}




























//
//RayRecordGMem RayMemory::GenerateRayPtrs(void* mem, size_t rayCount)
//{
//	RayRecordGMem record;
//
//	// Ray Size
//	size_t offset = 0;
//	size_t totalAlloc = TotalMemoryForRay(rayCount);
//	
//	// Pointer determination
//	byte* d_ptr = static_cast<byte*>(mem);	
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
//HitRecordGMem RayMemory::GenerateHitPtrs(void* mem, size_t rayCount)
//{
//	HitRecordGMem record;
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
//
//size_t RayMemory::TotalMemoryForRay(size_t rayCount)
//{
//	static_assert(sizeof(RayRecord) == sizeof(Vector4) + sizeof(Vec3AndUInt) * 2,
//				  "Ray record size sanity check.");
//	return sizeof(RayRecord) * rayCount;
//}
//
//size_t RayMemory::TotalMemoryForHit(size_t rayCount)
//{
//	static_assert(sizeof(::HitRecord) == sizeof(Vector3) + sizeof(unsigned int) * 2 + sizeof(float),
//				  "Hit record size sanity check.");
//	return sizeof(::HitRecord) * rayCount;
//}
//
//RayMemory::RayMemory()
//	: rayStackIn{nullptr, nullptr, nullptr}
//	, rayStackOut{nullptr, nullptr, nullptr}
//	, hitRecord{nullptr, nullptr}
//{}
//
//RayRecordGMem RayMemory::RayStackIn()
//{
//	return rayStackIn;
//}
//
//RayRecordGMem RayMemory::RayStackOut()
//{
//	return rayStackOut;
//}
//
//HitRecordGMem RayMemory::HitRecord()
//{
//	return hitRecord;
//}
//
//const ConstRayRecordGMem RayMemory::RayStackIn() const
//{
//	return rayStackIn;
//}
//
//const ConstRayRecordGMem RayMemory::RayStackOut() const
//{
//	return rayStackOut;
//}
//
//const ConstHitRecordGMem RayMemory::HitRecord() const
//{
//	return hitRecord;
//}
//
//void RayMemory::AllocForCameraRays(size_t rayCount)
//{
//	memRayIn = std::move(DeviceMemory(TotalMemoryForRay(rayCount)));
//	//memHit = std::move(DeviceMemory(TotalMemoryForHit(rayCount)));
//	rayStackIn = GenerateRayPtrs(memRayIn, rayCount);
//	//hitRecord = GenerateHitPtrs(memHit, rayCount);
//}
//
//MaterialRays RayMemory::ResizeRayIn(const MaterialRays& current,
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
//void RayMemory::ResizeRayOut(size_t rayCount)
//{
//	memRayOut = std::move(DeviceMemory(TotalMemoryForRay(rayCount)));
//	GenerateRayPtrs(memRayOut, rayCount);
//}
//
//void RayMemory::SwapRays(size_t rayCount)
//{
//	// Get Ready For Hit loop
//	memRayIn = std::move(memRayOut);
//	memHit = std::move(DeviceMemory(TotalMemoryForHit(rayCount)));
//	// Pointers
//	rayStackIn = rayStackOut;
//	GenerateHitPtrs(memHit, rayCount);
//}
//
//TracerCUDA::TracerCUDA()
//	: currentRayCount(0)
//	, parameters{}
//	, errorFunc(nullptr)
//{}
//
//TracerCUDA::~TracerCUDA()
//{}
//
//void TracerCUDA::Initialize(uint32_t seed)
//{
//	// Generate CUDA Tracer Interface
//	if(!CudaSystem::Initialize())
//	{
//		METU_ERROR_LOG("Unable to Init CUDA");
//	}
//	// TODO: remove this when multi-gpu
//	CUDA_CHECK(cudaSetDevice(0));
//
//	// Generate Random Seeds
//	rngMemory = RNGMemory(seed);
//
//	// Load background texture
//	ImageIO& imgIO = ImageIO::System();
//	Vector2ui size;
//	std::vector<Vector4> backgroundImage;
//	if(imgIO.ReadHDR(backgroundImage, size, "cape_hill_4k.hdr"))
//	{
//		// Load backgroundImage to texture
//		auto channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
//		CUDA_CHECK(cudaMallocArray(&texArray, &channelDesc, size[0], size[1]));
//		CUDA_CHECK(cudaMemcpyToArray(texArray, 0, 0,
//									 backgroundImage.data(),
//									 sizeof(Vector4) * size[0] * size[1],
//									 cudaMemcpyHostToDevice));
//		
//		// Texture descriptor
//		cudaTextureDesc td = {};
//		td.addressMode[0] = cudaAddressModeClamp;
//		td.addressMode[1] = cudaAddressModeClamp;
//		td.filterMode = cudaFilterModeLinear;
//		td.readMode = cudaReadModeElementType;
//		td.normalizedCoords = 1;
//
//		cudaResourceDesc rd = {};
//		rd.resType = cudaResourceTypeArray;
//		rd.res.array.array = texArray;
//
//		// Texture Object
//		if(backgroundTex != 0) CUDA_CHECK(cudaDestroyTextureObject(backgroundTex));
//		CUDA_CHECK(cudaCreateTextureObject(&backgroundTex,
//										   &rd,
//										   &td,
//										   nullptr));
//	}
//	else
//	{
//		assert(false);
//	}
//}
//
//void TracerCUDA::SetErrorCallback(ErrorCallbackFunction e)
//{
//	errorFunc = e;
//}
//
//void TracerCUDA::SetTime(double seconds)
//{
//	scene.ChangeTime(seconds);
//}
//
//void TracerCUDA::SetScene(const std::string& sceneFileName)
//{
//	// Check if file exists
//	SceneFile s;
//	IOError e = SceneFile::Load(s, sceneFileName);
//	if(e != IOError::OK)
//	{
//		if(errorFunc) errorFunc(Error{ErrorType::IO_ERROR, static_cast<uint32_t>(e)});
//	}
//	else
//	{
//		// TODO:
//		//Check availability of each file referenced by scene
//		scene = std::move(SceneGPU(s));
//	}
//}
//
//void TracerCUDA::SetParams(const TracerParameters& p)
//{
//	parameters = p;
//}
//
//void TracerCUDA::GenerateSceneAccelerator()
//{
//	// TODO:
//}
//
//void TracerCUDA::GenerateAccelerator(uint32_t objId)
//{
//	// TODO:
//}
//
//void TracerCUDA::ReportionImage(const Vector2ui& offset,
//								const Vector2ui& size)
//{
//	outputImage.ReportionImage(offset, size);
//}
//
//void TracerCUDA::ResizeImage(const Vector2ui& resolution)
//{
//	outputImage.ResizeImage(resolution);	
//}
//
//void TracerCUDA::ResetImage()
//{
//	outputImage.ResetImage();
//}
//
//std::vector<Vector3f> TracerCUDA::GetImage()
//{
//	return outputImage.GetImage();
//}
//
//void TracerCUDA::AssignAllMaterials()
//{
//	
//}
//
//void TracerCUDA::AssignMaterial(uint32_t matId)
//{
//
//}
//
//void TracerCUDA::LoadMaterial(uint32_t matId)
//{
//
//}
//
//void TracerCUDA::UnloadMaterial(uint32_t matId)
//{
//
//}
//
//void TracerCUDA::GenerateCameraRays(const CameraPerspective& camera,
//									const uint32_t samplePerPixel)
//{
//	// Total Ray Count and Allocation
//	uint32_t totalRayCount = samplePerPixel * samplePerPixel * 
//								outputImage.ImageSegment()[0] * 
//								outputImage.ImageSegment()[1];
//	rayMemory.AllocForCameraRays(totalRayCount);
//	sampleCount = samplePerPixel;
//	currentRayCount = totalRayCount;
//	
//	// TODO: segment ray generation to random gpus
//	// Single GPU Camera Ray Generation
//	const uint32_t gpuId = 0;
//
//	const uint32_t blockCount = CudaSystem::GPUList()[gpuId].RecommendedBlockCount();
//	const uint32_t threadCount = StaticThreadPerBlock1D;
//	const size_t sharedSize = rngMemory.SharedMemorySize(gpuId);
//
//	KCGenerateCameraRays<<<blockCount, threadCount, sharedSize>>>(rayMemory.RayStackIn(),
//																  rngMemory.RandomStack(gpuId),
//																  camera,
//																  samplePerPixel,
//																  outputImage.ImageResolution(),
//																  outputImage.ImageOffset(),
//																  outputImage.ImageSegment());
//	CUDA_KERNEL_CHECK();
//
//	//// DEBUG
//	//// Check Rays
//	//CUDA_CHECK(cudaDeviceSynchronize());
//
//	//for(int i = 0; i < totalRayCount; i++)
//	//{
//	//	Vector3 dir = rayMemory.RayStackIn().dirAndPixId[i].vec;
//	//	Vector3 pos = Vector3(rayMemory.RayStackIn().posAndMedium[i][0],
//	//						  rayMemory.RayStackIn().posAndMedium[i][1],
//	//						  rayMemory.RayStackIn().posAndMedium[i][2]);
//
//	//	METU_LOG("d(%f, %f, %f) p(%f, %f, %f)",
//	//			 dir[0], dir[1], dir[2],
//	//			 pos[0], pos[1], pos[2]);
//	//}
//}
//
//__device__
//bool SnapToInside(float& distance,
//
//				  const RayF& ray,
//				  const Vector3f liquidBottomLeft,
//				  const Vector3f liquidWorldLength)
//{
//	Vector3f pos;
//	return ray.IntersectsAABB(pos, distance,
//							  liquidBottomLeft,
//							  liquidBottomLeft + liquidWorldLength);
//}
//
//__global__
//void KernelVolumeLoop(RayRecordGMem gRays,
//					  RandomStackGMem gRand,
//					  // Volume Data
//					  SVODeviceData volumeSVO,
//					  VolumeDeviceData volumeDevice,
//					  FluidMaterialDeviceData material,
//					  // Output Image
//					  Vector3f* gImage,
//					  // Background Data
//					  cudaTextureObject_t backgroundTexture,
//					  // Limits
//					  uint32_t outgoingRayCount,
//					  uint32_t sampleCount)
//{
//	extern __shared__ uint32_t sRandState[];
//	RandomGPU rng(gRand.state, sRandState);
//
//	// Kernel Grid-Stride Loop
//	for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
//		threadId < outgoingRayCount;
//		threadId += (blockDim.x * gridDim.x))
//	{
//		uint32_t inRayId = threadId / 2;
//
//		RayRecord rRecord(gRays, inRayId);
//		HitRecord hRecord;
//
//		Vector3 totalRadiance = Zero3;
//		float totalOcclusion = 0.0f;
//
//		// Snap to water grid
//		const float distanceIncrement = 0.105f;
//		float distance = 0.0f;
//		if(SnapToInside(distance, rRecord.ray, volumeSVO.aabbMin, volumeDevice.worldSize))
//		{
//			rRecord.ray.AdvanceSelf(distance + 0.001f);			
//			Vector3 liquidTopRight = volumeSVO.aabbMax;
//			
//			int totalIterations = 0;
//			while((rRecord.ray.getPosition()[0] > volumeSVO.aabbMin[0] &&
//				   rRecord.ray.getPosition()[0] < liquidTopRight[0] &&
//
//				   rRecord.ray.getPosition()[1] > volumeSVO.aabbMin[1] &&
//				   rRecord.ray.getPosition()[1] < liquidTopRight[1] &&
//
//				   rRecord.ray.getPosition()[2] > volumeSVO.aabbMin[2] &&
//				   rRecord.ray.getPosition()[2] < liquidTopRight[2]) ||
//				  totalIterations < 50)
//			{
//				// Sample Texture
//				Vector3 delta = Vector3(1.0f) / static_cast<Vector3>(volumeDevice.size);
//				Vector3 normalizedCoords = (rRecord.ray.getPosition() - volumeSVO.aabbMin) / volumeDevice.worldSize;
//				normalizedCoords[1] = 1.0f - normalizedCoords[1];
//				float4 data = volumeDevice.volumeTex(normalizedCoords);
//		
//				// Data
//				float density = data.w / 15.0f;
//				Vector3f velocity;
//				velocity[0] = data.x;
//				velocity[1] = data.y;
//				velocity[2] = data.z;
//				
//				// Normal Approximation
//				Vector3 normal = Zero3;
//				UNROLL_LOOP
//				for(int i = 0; i < 8; i++)
//				{
//					Vector3i offset((i >> 0) & 0x1,
//									(i >> 1) & 0x1,
//									(i >> 2) & 0x1);
//					offset = (offset * 2) - Vector3i(1);					
//					Vector3 nearSample = normalizedCoords + static_cast<Vector3>(offset) * delta;
//					float density = volumeDevice.volumeTex(nearSample).w;
//
//					if(nearSample[0] < 0.0f || nearSample[0] > 1.0f ||
//					   nearSample[1] < 0.0f || nearSample[1] > 1.0f ||
//					   nearSample[2] < 0.0f || nearSample[2] > 1.0f)
//					{
//						density = 0.0f;
//					}
//					normal += static_cast<Vector3f>(offset) * density;
//				}
//				normal = -normal;
//				
//				// Beer Term (Absorbtion)
//				Vector3 beerFactor = Vector3(material.absorbtionCoeff * density);
//				float distance = distanceIncrement;
//				beerFactor[0] = expf(logf(beerFactor[0]) * distance);
//				beerFactor[1] = expf(logf(beerFactor[1]) * distance);
//				beerFactor[2] = expf(logf(beerFactor[2]) * distance);
//
//				// This means its on medium
//				// Cull unvalid normals (normal means the transition is very hard)
//				if(normal.Length() >= 0.7f)
//				{
//					normal.NormalizeSelf();
//
//					// Split
//					bool exitCase = rRecord.ray.getDirection().Dot(normal) > 0.0f;
//					Vector3 adjustedNormal = (exitCase) ? -normal : normal;
//					float hitMedium = (exitCase) ? 1.0f : material.ior;
//
//					// Total Reflection Case
//					Vector3 reflectionFactor = Vector3(1.0f);
//
//					RayF refractedRay(Zero3, Zero3);
//					bool refracted = rRecord.ray.Refract(refractedRay, adjustedNormal,
//														 rRecord.medium, hitMedium);
//					if(refracted)
//					{
//						// Fresnel Term
//						// Schclick's Approx
//						float cosTetha = (exitCase)
//							? -normal.Dot(refractedRay.getDirection())
//							: -normal.Dot(rRecord.ray.getDirection());
//						float r0 = (rRecord.medium - hitMedium) / (rRecord.medium + hitMedium);
//						float cosTerm = 1.0f - cosTetha;
//						cosTerm = cosTerm * cosTerm * cosTerm * cosTerm * cosTerm;
//						r0 = r0 * r0;
//						float fresnel = r0 + (1.0f - r0) * cosTerm;
//
//						beerFactor = (exitCase) ? Vector3(1.0f) : beerFactor;
//
//						// Energy Factors
//						Vector3 refractionFactor = Vector3(1.0f - fresnel) * beerFactor;
//						reflectionFactor = Vector3(fresnel) * beerFactor;
//
//						// Write this
//						refractedRay.NormalizeDirSelf();
//						refractedRay.AdvanceSelf(hRecord.distance + MathConstants::Epsilon);
//
//						// Record Save
//						if(threadId % 2 == 0)
//							rRecord.ray = refractedRay;
//						totalRadiance *= refractionFactor;
//
//						//refractRecord.ray = refractedRay;
//						//refractRecord.medium = hitMedium;
//						//refractRecord.pixelId = rRecord.pixelId;
//						//refractRecord.sampleId = rRecord.sampleId;
//						//refractRecord.totalRadiance = rRecord.totalRadiance * refractionFactor;
//					}
//					// Reflection
//					RayF reflectedRay = rRecord.ray.Reflect(adjustedNormal);
//
//					// Write this
//					reflectedRay.NormalizeDirSelf();
//					reflectedRay.AdvanceSelf(hRecord.distance + MathConstants::Epsilon);
//
//					if(threadId % 2 == 1)
//						rRecord.ray = reflectedRay;
//					totalRadiance *= reflectionFactor;
//
//					//refractRecord.ray = reflectedRay;
//					//refractRecord.medium = rRecord.medium;
//					//refractRecord.pixelId = rRecord.pixelId;
//					//refractRecord.sampleId = rRecord.sampleId;
//					//refractRecord.totalRadiance = rRecord.totalRadiance * reflectionFactor;
//				}
//				
//				// Ray hits!
//				// Do stuff
//				//Vector3 waterColor = Vector3(0.0f, 0.0f, 1.0f);
//				//totalRadiance += density * waterColor * (1.0f - totalOcclusion);
//				//totalOcclusion += density * (1.0f - totalOcclusion);
//		
//				// Advance ray
//				rRecord.ray.AdvanceSelf(distanceIncrement);
//				totalIterations++;
//			}
//		}
//		// We hit outside generate UV coordinates for spherical
//		Vector3 relativeCoord = rRecord.ray.getDirection();
//		float v = relativeCoord[1] * 0.5f + 0.5f;
//		float u = std::atan2(relativeCoord[2], relativeCoord[0]) * 0.5f * MathConstants::InvPI;
//		u += 0.5f;
//
//		// Fetch background texture
//		float4 texSample = tex2D<float4>(backgroundTexture, u, v);
//		Vector3 texSampVector = Vector3(texSample.x, texSample.y, texSample.z);
//
//		//printf("%f %f %f\n", texSampVector[0], texSampVector[1], texSampVector[2]);
//		totalRadiance += texSampVector * (1.0f - totalOcclusion);
//
//		// We are out of location
//		totalRadiance /= (sampleCount * sampleCount) * 2;
//
//		// Write to pixel
//		float* gPixel = reinterpret_cast<float*>(gImage + rRecord.pixelId);
//		atomicAdd(gPixel + 0, totalRadiance[0]);
//		atomicAdd(gPixel + 1, totalRadiance[1]);
//		atomicAdd(gPixel + 2, totalRadiance[2]);
//	}
//}
//
//void TracerCUDA::HitRays(int frame)
//{
//	// Super Inlined code for project
//	// Hit and bounce on same kernel
//	// Data etc	
//	VolumeGPU& vol = static_cast<VolumeGPU&>(scene.Volume(0));
//	vol.ChangeFrame(static_cast<double>(frame));
//	CUDA_CHECK(cudaDeviceSynchronize());
//	const SVODevice& svo = vol.SVO();
//	const FluidMaterialGPU& material = static_cast<const FluidMaterialGPU&>(scene.Material(0));
//
//	// Kernel Call
//	const uint32_t gpuId = 0;
//	const uint32_t blockCount = CudaSystem::GPUList()[gpuId].RecommendedBlockCount();
//	const uint32_t threadCount = StaticThreadPerBlock1D;
//	const size_t sharedSize = rngMemory.SharedMemorySize(gpuId);
//	
//	KernelVolumeLoop<<<blockCount, threadCount, sharedSize>>>(rayMemory.RayStackIn(),
//															  rngMemory.RandomStack(gpuId),
//															  
//															  svo,
//															  vol,
//															  material,
//															  
//															  outputImage.ImageGMem(),
//
//															  backgroundTex,
//
//															  currentRayCount * 2,
//															  sampleCount);
//	CUDA_KERNEL_CHECK();
//	
//	CUDA_CHECK(cudaDeviceSynchronize());
//
//	auto image = outputImage.GetImage();
//	ImageIO::System().WriteAsPNG(image.data(), outputImage.ImageSegment(), std::to_string(frame) + ".png");
//	CUDA_CHECK(cudaDeviceSynchronize());
//	currentRayCount = 0;
//
//	// TODO:
//	// TODO:
//	// After hit determine remaining ray count
//}
//
//void TracerCUDA::GetMaterialRays(RayRecordCPU&, HitRecordCPU&,
//								 uint32_t rayCount, uint32_t matId)
//{
//	// TODO: Adjust memory
//}
//
//void TracerCUDA::AddMaterialRays(const RayRecordCPU&, const HitRecordCPU&,
//								 uint32_t rayCount, uint32_t matId)
//{
//	// TODO: Adjust memory
//}
//
//void TracerCUDA::BounceRays()
//{
//	//// First determine potential output ray count
//	//RayCount
//	//rayMemory.ResizeRayOut(2 * currentRayCount);
//
//	
//	// TODO::
//
//	// Rays are sorted and ready for material calls
//	// For each loaded material launch
//
//}
//
//uint32_t TracerCUDA::RayCount()
//{
//	return currentRayCount;
//}
//
////// DELETE THOSE
////void TracerCUDA::LoadBackgroundCubeMap(const std::vector<float>& cubemap)
////{
////
////}
////
////void TracerCUDA::LoadFluidToGPU(const std::vector<float>& velocityDensity,
////								const Vector3ui& size)
////{
////	// Copy Density
////	velocityDensityTexture.Copy(reinterpret_cast<const byte*>(velocityDensity.data()), size);
////}
////
////void TracerCUDA::CS568GenerateCameraRays(const CameraPerspective& cam,
////										 const Vector2ui resolution,
////										 const uint32_t samplePerPixel)
////{
////	totalRayCount = resolution[0] * resolution[1] * samplePerPixel * samplePerPixel;
////
////	AllocateRayStack(resolution[0] * resolution[1] * samplePerPixel * samplePerPixel);
////	AllocateRandomStack();
////	AllocateImage(resolution);
////
////	uint32_t blockCount = CudaSystem::GPUList()[0].RecommendedBlockCount();
////	uint32_t threadCount = StaticThreadPerBlock1D;
////	uint32_t sharedSize = sizeof(uint32_t) * StaticThreadPerBlock1D;
////
////	KCGenerateCameraRays<<<blockCount, threadCount, sharedSize>>>(rayStackIn,
////																  random,
////																  cam,
////																  samplePerPixel,
////																  resolution,
////																  Vector2ui(0, 0),
////																  resolution);
////	CUDA_KERNEL_CHECK();
////
////	//// DEBUG
////	//// Check Rays
////	//CUDA_CHECK(cudaDeviceSynchronize());
////
////	//for(int i = 0; i < totalRayCount; i++)
////	//{
////	//	Vector3 dir = rayStackIn.dirAndPixId[i].vec;
////	//	Vector3 pos = Vector3(rayStackIn.posAndMedium[i][0], 
////	//						  rayStackIn.posAndMedium[i][1], 
////	//						  rayStackIn.posAndMedium[i][2]);
////
////	//	METU_LOG("d(%f, %f, %f) p(%f, %f, %f)",
////	//			 dir[0], dir[1], dir[2],
////	//			 pos[0], pos[1], pos[2]);
////	//}
////}
//
//
////__device__ bool SnapToInside(float& distance,
////
////							 const RayF& ray,
////							 const Vector3f liquidBottomLeft,
////							 const Vector3f liquidWorldLength)
////{
////	Vector3f pos;
////	distance = ray.IntersectsAABB(pos,
////								  liquidBottomLeft,
////								  liquidBottomLeft + liquidWorldLength);
////	
////	if(distance == FLT_MAX)
////		return false;
////	return true;
////}
////
////__device__ bool CheckIntersection(float& distance,
////								  float& density,
////								  Vector3f& velocity,
////								  
////								  const RayF& ray,
////
////								  const Texture3<float4>& liquidTexture,
////								  const Vector3ui& liquidDim,
////								  const Vector3f& liquidBottomLeft,
////								  const Vector3f& liquidWorldLength)
////{
////	Vector3 gridSpan = liquidWorldLength / Vector3f(liquidDim[0],
////													liquidDim[1],
////													liquidDim[2]);
////	Vector3 gridOffset = (((ray.getPosition() - liquidBottomLeft) / gridSpan).FloorSelf()) * gridSpan;
////	Vector3 gridWorld = liquidBottomLeft + gridOffset;
////	
////	// Calculate next distance
////	distance = IntersectDistance(ray,
////								 gridWorld,
////								 gridWorld + gridSpan);
////
////	// Sample Texture
////	Vector3 normalizedCoords = (ray.getPosition() - liquidBottomLeft) / Vector3f(liquidDim[0],
////																				 liquidDim[1],
////																				 liquidDim[2]);
////	float4 data = liquidTexture(normalizedCoords);
////
////	// Push Texture
////	density = data.w;
////	velocity[0] = data.x;
////	velocity[1] = data.y;
////	velocity[2] = data.z;
////
////	// FLT max means no intersection
////	if(distance == FLT_MAX)
////		return false;
////	return true;
////}
//
////// Delete this kernel
////__global__ void KernelLoop(Vector3f* outImage,
////
////						   ConstRayRecordGMem gRays,
////						   RandomStackGMem gRand,
////						   const uint32_t totalRayCount,
////
////						   const Vector3f backgroundColor,
////
////						   // Texture Related
////						   Texture3<float4> liquidTexture,
////						   const Vector3ui liquidDim,
////						   const Vector3f liquidBottomLeft,
////						   const Vector3f liquidWorldLength)
////{
////	extern __shared__ uint32_t sRandState[];
////	RandomGPU rng(gRand.state, sRandState);
////
////	const uint32_t totalWorkCount = totalRayCount;
////
////	// Kernel Grid-Stride Loop
////	for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
////		threadId < totalWorkCount;
////		threadId += (blockDim.x * gridDim.x))
////	{
////		// Load Ray to registers
////		RayRecord rayData(gRays, threadId);
////		rayData.totalRadiance = Vector3(1.0f);
////
////		// Accumulate
////		Vector3f totalIllumination = Zero3;
////		float totalOcclusion = 0.0f;
////		
////		// Loop until ray hits
////		RayF r = rayData.ray;
////		
////		Vector3 waterColor = Vector3(0.0f, 0.0f, 1.0f);
////
////		// Snap to water grid
////		float distance = 0.0f;
////		if(SnapToInside(distance, r, liquidBottomLeft, liquidWorldLength))
////		{
////			//if(threadId == (512 * 256 + 256))
////			//	printf("(%f, %f, %f)\n",
////			//		   r.getPosition()[0],
////			//		   r.getPosition()[1],
////			//		   r.getPosition()[2]);
////
////			r.AdvanceSelf(distance + 0.001f);
////
////
////			Vector3 liquidTopRight = liquidBottomLeft + liquidWorldLength;
////
////			if(threadId == (512 * 256 + 256))
////			{
////				//printf("(%f, %f, %f)\n",
////				//	   r.getPosition()[0],
////				//	   r.getPosition()[1],
////				//	   r.getPosition()[2]);
////
////				//printf("(%f, %f, %f)\n",
////				//	  liquidBottomLeft[0],
////				//	  liquidBottomLeft[1],
////				//	  liquidBottomLeft[2]);
////
////				//printf("(%f, %f, %f)\n",
////				//	   liquidTopRight[0],
////				//	   liquidTopRight[1],
////				//	   liquidTopRight[2]);
////			}
////
////			while(r.getPosition()[0] > liquidBottomLeft[0] && 
////					r.getPosition()[0] < liquidTopRight[0] &&
////
////					r.getPosition()[1] > liquidBottomLeft[1] &&
////					r.getPosition()[1] < liquidTopRight[1] &&
////
////					r.getPosition()[2] > liquidBottomLeft[2] &&
////					r.getPosition()[2] < liquidTopRight[2])
////			{
////
////				//if(threadId == (512 * 256 + 256))
////				//   printf("looping\n");
////
////				// Sample Texture
////				Vector3 normalizedCoords = (r.getPosition() - liquidBottomLeft) / Vector3f(liquidWorldLength[0],
////																						   liquidWorldLength[1],
////																						   liquidWorldLength[2]);
////				float4 data = liquidTexture(normalizedCoords);
////
////				//if(threadId == (512 * 256 + 256))
////				//{
////				//	printf("(%f, %f, %f)\n",
////				//		   r.getPosition()[0],
////				//		   r.getPosition()[1],
////				//		   r.getPosition()[2]);
////				//	printf("(%f, %f, %f)\n",
////				//		   normalizedCoords[0],
////				//		   normalizedCoords[1],
////				//		   normalizedCoords[2]);
////				//	printf("%f\n", data.w);
////				//}
////
////				float density = fminf(data.w, 1.0f);
////				Vector3f velocity;
////				velocity[0] = data.x;
////				velocity[1] = data.y;
////				velocity[2] = data.z;
////
////				// Ray hits!
////				// Do stuff
////				totalIllumination += density * waterColor * (1.0f - totalOcclusion);
////				totalOcclusion += density * (1.0f - totalOcclusion);
////
////				// Advance ray
////				Vector3 gridSpan = liquidWorldLength / Vector3f(liquidDim[0],
////																liquidDim[1],
////																liquidDim[2]);
////				float spanAvg = gridSpan[0] + gridSpan[1] + gridSpan[2] * 0.33f;
////				//r.AdvanceSelf(spanAvg * 1.732);
////				r.AdvanceSelf(0.05f);
////			}
////
////		}
////
////		totalIllumination += backgroundColor * (1.0f - totalOcclusion);
////
////		// Write total illumination
////		float* pixelAdress = reinterpret_cast<float*>(outImage + rayData.pixelId);
////		atomicAdd(pixelAdress + 0, totalIllumination[0]);
////		atomicAdd(pixelAdress + 1, totalIllumination[1]);
////		atomicAdd(pixelAdress + 2, totalIllumination[2]);		
////	}
////}
////
////void TracerCUDA::LaunchRays(const Vector3f& backgroundColor,
////							const Vector3ui& textureSize,
////							const Vector3f& bottomLeft,
////							const Vector3f& length)
////{
////	uint32_t blockCount = CudaSystem::GPUList()[0].RecommendedBlockCount();
////	uint32_t threadCount = StaticThreadPerBlock1D;
////	uint32_t sharedSize = sizeof(uint32_t) * StaticThreadPerBlock1D;
////
////	KernelLoop<<<blockCount, threadCount, sharedSize>>>(dOutImage, rayStackIn, random,
////													    totalRayCount,
////													    backgroundColor,
////													    velocityDensityTexture,
////													    textureSize,
////													    bottomLeft,
////													    length);
////	CUDA_KERNEL_CHECK();
////}
////
////std::vector<Vector3> TracerCUDA::GetImage(const Vector2ui& resolution)
////{
////	CUDA_CHECK(cudaDeviceSynchronize());
////	size_t pixelCount = resolution[0] * resolution[1];
////	std::vector<Vector3> out(pixelCount);
////	memcpy(out.data(), dOutImage, sizeof(Vector3f) * pixelCount);
////		
////	return out;
////}
