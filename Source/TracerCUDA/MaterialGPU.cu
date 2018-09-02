#include "MaterialGPU.cuh"
//#include "RayLib/RayHitStructs.h"
#include "RayLib/CudaConstants.h"
#include "RayLib/Error.h"
#include "RayLib/Random.cuh"
#include "RayLib/Constants.h"

#include "SurfaceDeviceI.cuh"

//
//template <class DataFetchFunc, class T>
//__global__ 
//void KCBounceRays(// Outgoing Rays
//				  RayRecordGMem gOutRays,
//				  // Incoming Rays
//				  const ConstHitRecordGMem gHits,
//				  const ConstRayRecordGMem gRays,
//				  const FluidMaterialDeviceData materialData,
//				  // SM Data
//				  RandomStackGMem gRand,
//				  // Surface Data
//				  const Vector2ui* gSurfaceIndexList,
//				  const T* gSurfaces,
//				  // Limits
//				  uint32_t rayCount)
//{
//	extern __shared__ uint32_t sRandState[];
//	RandomGPU rng(gRand.state, sRandState);
//
//	// Kernel Grid-Stride Loop
//	for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
//		threadId < rayCount;
//		threadId += (blockDim.x * gridDim.x))
//	{
//		// Input
//		HitRecord hRecord(gHits, threadId);
//		RayRecord rRecord(gRays, threadId);
//		// Output
//		RayRecord outRecord[2]
//
//		// Actual Call
//		materialData.Bounce<DataFetchFunc, T>
//		(
//			outRecord,			
//			hRecord, 
//			rRecord,			
//		);
//		
//		// Write
//		if(outRecord[0].medium != 0.0f)
//		{
//			gOutRays.Save(refractRecord, threadId);
//		}
//		if(outRecord[1].medium != 0.0f)
//		{
//			gOutRays.Save(reflectRecord, threadId + rayCount);
//		}		
//	}
//}

FluidMaterialGPU::FluidMaterialGPU(uint32_t materialId,
								   float indexOfRefraction,
								   // Color Interpolation
								   const std::vector<Vector3f>& color,
								   const std::vector<float>& colorInterp,
								   // Opacity Interpolation
								   const std::vector<float>& opacity,
								   const std::vector<float>& opacityInterp,
								   // Transparency
								   Vector3f transparency,
								   // Volumetric Parameters
								   float absorbtionCoeff,
								   float scatteringCoeff)
	: color(std::move(color))
	, colorInterp(std::move(colorInterp))
	, opacity(std::move(opacity))
	, opacityInterp(std::move(opacityInterp))
	, FluidMaterialDeviceData 
		{ 
			nullptr, nullptr, nullptr, nullptr, 
			transparency, absorbtionCoeff, 
			scatteringCoeff, indexOfRefraction 
		}
{}

uint32_t FluidMaterialGPU::Id() const
{
	return materialId;
}

//void FluidMaterialGPU::BounceRays(// Outgoing Rays
//								  RayRecordGMem gOutRays,
//								  // Incoming Rays
//								  const ConstHitRecordGMem gHits,
//								  const ConstRayRecordGMem gRays,
//								  // Limits
//								  uint64_t rayCount,
//								  // Surfaces
//								  const Vector2ui* gSurfaceIndexList,
//								  const void* gSurfaces,
//								  SurfaceType t)
//{
//
//	//.... 
//	//CudaSystem::GPUCallX(CudaSystem::CURRENT_DEVICE, 0, 0, 0);
//}

Error FluidMaterialGPU::Load()
{
	//Determine total
	size_t totalSize = color.size() * sizeof(Vector3f) +
					   colorInterp.size() * sizeof(float) +
					   opacity.size() * sizeof(float) +
					   opacityInterp.size() * sizeof(float);
	mem = std::move(DeviceMemory(totalSize));
	char* memPtr = static_cast<char*>(mem);

	// Pointers
	size_t offset = 0;
	gColor = reinterpret_cast<Vector3f*>(memPtr + offset);
	offset += color.size() * sizeof(Vector3f);
	gColorInterp = reinterpret_cast<float*>(memPtr + offset);
	offset += colorInterp.size() * sizeof(float);
	gOpacity = reinterpret_cast<float*>(memPtr + offset);
	offset += opacity.size() * sizeof(float);
	gOpacityInterp = reinterpret_cast<float*>(memPtr + offset);
	offset += opacityInterp.size() * sizeof(float);
	assert(totalSize == offset);

	return Error{ErrorType::ANY_ERROR, Error::OK};
}

void FluidMaterialGPU::Unload()
{
	mem = std::move(DeviceMemory());
	gColor = nullptr;
	gColorInterp = nullptr;
	gOpacity = nullptr;
	gOpacityInterp = nullptr;
}