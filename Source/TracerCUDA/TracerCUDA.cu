#include <random>

#include "TracerCUDA.h"
#include "CameraKernels.cuh"
#include "RayLib/Camera.h"
#include "RayLib/CudaConstants.h"
#include "RayLib/Random.cuh"

#include "RayLib/Log.h"

void TracerCUDA::AllocateRayStack(size_t count)
{
	// Ray Size
	size_t raySize = sizeof(Vector4) + sizeof(Vec3AndUInt) * 2;
	size_t totalAlloc = raySize * count * 2;

	// Actual Allocation
	commonMemory.emplace_back(totalAlloc);
	byte* d_ptr = static_cast<byte*>(commonMemory.back());

	// Pointer determination
	size_t offset = 0;
	rayStackIn.posAndMedium = reinterpret_cast<Vector4*>(d_ptr + offset);
	offset += count * sizeof(Vector4);
	rayStackIn.dirAndPixId = reinterpret_cast<Vec3AndUInt*>(d_ptr + offset);
	offset += count * sizeof(Vec3AndUInt);
	rayStackIn.radAndSampId = reinterpret_cast<Vec3AndUInt*>(d_ptr + offset);
	offset += count * sizeof(Vec3AndUInt);

	rayStackOut.posAndMedium = reinterpret_cast<Vector4*>(d_ptr + offset);
	offset += count * sizeof(Vector4);
	rayStackOut.dirAndPixId = reinterpret_cast<Vec3AndUInt*>(d_ptr + offset);
	offset += count * sizeof(Vec3AndUInt);
	rayStackOut.radAndSampId = reinterpret_cast<Vec3AndUInt*>(d_ptr + offset);
	offset += count * sizeof(Vec3AndUInt);

	// Check that we did properly
	assert(offset == totalAlloc);
}

void TracerCUDA::AllocateImage(Vector2ui resolution)
{
	commonMemory.emplace_back(sizeof(Vector3f) * resolution[0] * resolution[1]);
	dOutImage = static_cast<Vector3f*>(commonMemory.back());
}

void TracerCUDA::AllocateRandomStack()
{
	std::mt19937 rng;
	rng.seed(999);
	
	// WRONG SINGLE GPU IMPLEMENTATION
	auto gpu = CudaSystem::GPUList()[0];
	size_t totalCount = gpu.RecommendedBlockCount() * StaticThreadPerBlock1D;
	size_t totalSize = totalCount * sizeof(uint32_t);

	// Actual Allocation
	commonMemory.emplace_back(totalSize);
	byte* d_ptr = static_cast<byte*>(commonMemory.back());
	random.state = reinterpret_cast<uint32_t*>(d_ptr);

	// Init
	std::vector<uint32_t> seeds(totalCount);
	for(size_t i = 0; i < totalCount; i++)
	{
		random.state[i] = rng();
	}
}

void TracerCUDA::THRDAssignScene(const SceneI& )
{

}

void TracerCUDA::THRDSetParams(const TracerParameters&)
{

}

void TracerCUDA::THRDGenerateSceneAccelerator()
{

}

void TracerCUDA::THRDGenerateAccelerator(uint32_t objId)
{

}

void TracerCUDA::THRDAssignImageSegment(const Vector2ui& pixelStart,
										const Vector2ui& pixelEnd)
{

}

void TracerCUDA::THRDAssignAllMaterials()
{

}

void TracerCUDA::THRDAssignMaterial(uint32_t matId)
{

}

void TracerCUDA::THRDLoadMaterial(uint32_t matId)
{

}

void TracerCUDA::THRDUnloadMaterial(uint32_t matId)
{

}

void TracerCUDA::THRDGenerateCameraRays(const CameraPerspective& camera,
										const uint32_t samplePerPixel)
{

}

void TracerCUDA::THRDHitRays()
{

}

void TracerCUDA::THRDGetMaterialRays(const RayRecodCPU&, uint32_t rayCount, uint32_t matId)
{

}

void TracerCUDA::THRDAddMaterialRays(const ConstRayRecodCPU&, uint32_t rayCount, uint32_t matId)
{

}

void TracerCUDA::THRDBounceRays()
{

}

uint32_t TracerCUDA::THRDRayCount()
{
	return 0;
}

TracerCUDA::TracerCUDA()
	: velocityDensityTexture(InterpolationType::LINEAR,
					  EdgeResolveType::CLAMP,
					  false,
					  Vector3ui(128, 128, 128))
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
}

// DELETE THOSE
void TracerCUDA::LoadBackgroundCubeMap(const std::vector<float>& cubemap)
{

}

void TracerCUDA::LoadFluidToGPU(const std::vector<float>& velocityDensity,
								const Vector3ui& size)
{
	// Copy Density
	velocityDensityTexture.Copy(reinterpret_cast<const byte*>(velocityDensity.data()), size);
}

void TracerCUDA::CS568GenerateCameraRays(const CameraPerspective& cam,
										 const Vector2ui resolution,
										 const uint32_t samplePerPixel)
{
	totalRayCount = resolution[0] * resolution[1] * samplePerPixel * samplePerPixel;

	AllocateRayStack(resolution[0] * resolution[1] * samplePerPixel * samplePerPixel);
	AllocateRandomStack();
	AllocateImage(resolution);

	uint32_t blockCount = CudaSystem::GPUList()[0].RecommendedBlockCount();
	uint32_t threadCount = StaticThreadPerBlock1D;
	uint32_t sharedSize = sizeof(uint32_t) * StaticThreadPerBlock1D;

	KCGenerateCameraRays<<<blockCount, threadCount, sharedSize>>>(rayStackIn,
																  random,
																  cam,
																  samplePerPixel,
																  resolution,
																  Vector2ui(0, 0),
																  resolution);
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

__device__ float IntersectDistance(const RayF& ray,

								   const Vector3f& min,
								   const Vector3f& max)
{
	Vector3 invD = Vector3(1.0f) / ray.getDirection();
	Vector3 t0 = (min - ray.getPosition()) * invD;
	Vector3 t1 = (max - ray.getPosition()) * invD;

	float tMin = -std::numeric_limits<float>::max();
	float tMax = std::numeric_limits<float>::max();
	float t = std::numeric_limits<float>::max();

	#pragma unroll
	for(int i = 0; i < 3; i++)
	{
		tMin = std::max(tMin, std::min(t0[i], t1[i]));
		tMax = std::min(tMax, std::max(t0[i], t1[i]));
		t = (t0[i] > 0.0f) ? std::min(t, t0[i]) : t;
		t = (t1[i] > 0.0f) ? std::min(t, t1[i]) : t;
	}


	if(tMax >= tMin)
		return t;
	else
		return FLT_MAX;
}


__device__ bool SnapToInside(float& distance,

							 const RayF& ray,
							 const Vector3f liquidBottomLeft,
							 const Vector3f liquidWorldLength)
{
	distance = IntersectDistance(ray,
								 liquidBottomLeft,
								 liquidBottomLeft + liquidWorldLength);
	
	if(distance == FLT_MAX)
		return false;
	return true;
}

__device__ bool CheckIntersection(float& distance,
								  float& density,
								  Vector3f& velocity,
								  
								  const RayF& ray,

								  const Texture3<float4>& liquidTexture,
								  const Vector3ui& liquidDim,
								  const Vector3f& liquidBottomLeft,
								  const Vector3f& liquidWorldLength)
{
	Vector3 gridSpan = liquidWorldLength / Vector3f(liquidDim[0],
													liquidDim[1],
													liquidDim[2]);
	Vector3 gridOffset = (((ray.getPosition() - liquidBottomLeft) / gridSpan).FloorSelf()) * gridSpan;
	Vector3 gridWorld = liquidBottomLeft + gridOffset;
	
	// Calculate next distance
	distance = IntersectDistance(ray,
								 gridWorld,
								 gridWorld + gridSpan);

	// Sample Texture
	Vector3 normalizedCoords = (ray.getPosition() - liquidBottomLeft) / Vector3f(liquidDim[0],
																				 liquidDim[1],
																				 liquidDim[2]);
	float4 data = liquidTexture(normalizedCoords);

	// Push Texture
	density = data.w;
	velocity[0] = data.x;
	velocity[1] = data.y;
	velocity[2] = data.z;

	// FLT max means no intersection
	if(distance == FLT_MAX)
		return false;
	return true;
}

// Delete this kernel
__global__ void KernelLoop(Vector3f* outImage,

						   ConstRayStackGMem gRays,
						   RandomStackGMem gRand,
						   const uint32_t totalRayCount,

						   const Vector3f backgroundColor,

						   // Texture Related
						   Texture3<float4> liquidTexture,
						   const Vector3ui liquidDim,
						   const Vector3f liquidBottomLeft,
						   const Vector3f liquidWorldLength)
{
	extern __shared__ uint32_t sRandState[];
	RandomGPU rng(gRand.state, sRandState);

	const uint32_t totalWorkCount = totalRayCount;

	// Kernel Grid-Stride Loop
	for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
		threadId < totalWorkCount;
		threadId += (blockDim.x * gridDim.x))
	{
		// Load Ray to registers
		RayStack rayData(gRays, threadId);
		rayData.totalRadiance = Vector3(1.0f);

		// Accumulate
		Vector3f totalIllumination = Zero3;
		float totalOcclusion = 0.0f;
		
		// Loop until ray hits
		RayF r = rayData.ray;
		
		Vector3 waterColor = Vector3(0.0f, 0.0f, 1.0f);

		// Snap to water grid
		float distance = 0.0f;
		if(SnapToInside(distance, r, liquidBottomLeft, liquidWorldLength))
		{
			//if(threadId == (512 * 256 + 256))
			//	printf("(%f, %f, %f)\n",
			//		   r.getPosition()[0],
			//		   r.getPosition()[1],
			//		   r.getPosition()[2]);

			r.AdvanceSelf(distance + 0.001f);


			Vector3 liquidTopRight = liquidBottomLeft + liquidWorldLength;

			if(threadId == (512 * 256 + 256))
			{
				//printf("(%f, %f, %f)\n",
				//	   r.getPosition()[0],
				//	   r.getPosition()[1],
				//	   r.getPosition()[2]);

				//printf("(%f, %f, %f)\n",
				//	  liquidBottomLeft[0],
				//	  liquidBottomLeft[1],
				//	  liquidBottomLeft[2]);

				//printf("(%f, %f, %f)\n",
				//	   liquidTopRight[0],
				//	   liquidTopRight[1],
				//	   liquidTopRight[2]);
			}

			while(r.getPosition()[0] > liquidBottomLeft[0] && 
					r.getPosition()[0] < liquidTopRight[0] &&

					r.getPosition()[1] > liquidBottomLeft[1] &&
					r.getPosition()[1] < liquidTopRight[1] &&

					r.getPosition()[2] > liquidBottomLeft[2] &&
					r.getPosition()[2] < liquidTopRight[2])
			{

				//if(threadId == (512 * 256 + 256))
				//   printf("looping\n");

				// Sample Texture
				Vector3 normalizedCoords = (r.getPosition() - liquidBottomLeft) / Vector3f(liquidWorldLength[0],
																						   liquidWorldLength[1],
																						   liquidWorldLength[2]);
				float4 data = liquidTexture(normalizedCoords);

				//if(threadId == (512 * 256 + 256))
				//{
				//	printf("(%f, %f, %f)\n",
				//		   r.getPosition()[0],
				//		   r.getPosition()[1],
				//		   r.getPosition()[2]);
				//	printf("(%f, %f, %f)\n",
				//		   normalizedCoords[0],
				//		   normalizedCoords[1],
				//		   normalizedCoords[2]);
				//	printf("%f\n", data.w);
				//}

				float density = fminf(data.w, 1.0f);
				Vector3f velocity;
				velocity[0] = data.x;
				velocity[1] = data.y;
				velocity[2] = data.z;

				// Ray hits!
				// Do stuff
				totalIllumination += density * waterColor * (1.0f - totalOcclusion);
				totalOcclusion += density * (1.0f - totalOcclusion);

				// Advance ray
				Vector3 gridSpan = liquidWorldLength / Vector3f(liquidDim[0],
																liquidDim[1],
																liquidDim[2]);
				float spanAvg = gridSpan[0] + gridSpan[1] + gridSpan[2] * 0.33f;
				//r.AdvanceSelf(spanAvg * 1.732);
				r.AdvanceSelf(0.05f);
			}

		}

		totalIllumination += backgroundColor * (1.0f - totalOcclusion);

		// Write total illumination
		float* pixelAdress = reinterpret_cast<float*>(outImage + rayData.pixelId);
		atomicAdd(pixelAdress + 0, totalIllumination[0]);
		atomicAdd(pixelAdress + 1, totalIllumination[1]);
		atomicAdd(pixelAdress + 2, totalIllumination[2]);		
	}
}

void TracerCUDA::LaunchRays(const Vector3f& backgroundColor,
							const Vector3ui& textureSize,
							const Vector3f& bottomLeft,
							const Vector3f& length)
{
	uint32_t blockCount = CudaSystem::GPUList()[0].RecommendedBlockCount();
	uint32_t threadCount = StaticThreadPerBlock1D;
	uint32_t sharedSize = sizeof(uint32_t) * StaticThreadPerBlock1D;

	KernelLoop<<<blockCount, threadCount, sharedSize>>>(dOutImage, rayStackIn, random,
													    totalRayCount,
													    backgroundColor,
													    velocityDensityTexture,
													    textureSize,
													    bottomLeft,
													    length);
	CUDA_KERNEL_CHECK();
}

std::vector<Vector3> TracerCUDA::GetImage(const Vector2ui& resolution)
{
	CUDA_CHECK(cudaDeviceSynchronize());
	size_t pixelCount = resolution[0] * resolution[1];
	std::vector<Vector3> out(pixelCount);
	memcpy(out.data(), dOutImage, sizeof(Vector3f) * pixelCount);
		
	return out;
}
