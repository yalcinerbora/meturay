#include "TracerLogics.cuh"

#include "RayLib/TracerError.h"
#include "TracerLib/GPUScene.h"
#include "TracerLib/RayMemory.h"

TracerBasic::TracerBasic(GPUBaseAcceleratorI& baseAccelerator,
						 const AcceleratorBatchMappings& a,
						 const MaterialBatchMappings& m,
						 const TracerParameters& options,
						 uint32_t hitStructSize,
						 const Vector2i maxMats,
						 const Vector2i maxAccels)
	: TracerBaseLogic(baseAccelerator, 
					  a, m, 
					  options, 
					  initals,
					  hitStructSize,
					  maxMats,
					  maxAccels)
{}

TracerError TracerBasic::Initialize()
{
	return TracerError::OK;
}

size_t TracerBasic::GenerateRays(RayMemory& rayMem, RNGMemory& rngMem,
								 const GPUScene& scene,
								 int cameraId,
								 int samplePerLocation,
								 Vector2i resolution,
								 Vector2i pixelStart,
								 Vector2i pixelEnd)
{
	pixelEnd = Vector2i::Min(resolution, pixelEnd);
	Vector2i pixelCount = (pixelEnd - pixelStart);
	size_t currentRayCount = pixelCount[0] * samplePerLocation *
							 pixelCount[1] * samplePerLocation;
	CameraPerspective currentCam = scene.CamerasCPU()[cameraId];

	// Allocate enough space for ray
	rayMem.ResizeRayOut(currentRayCount, PerRayAuxDataSize());

	
	// Basic Tracer does classic camera to light tracing
	// Thus its initial rays are from camera

	// Call multi-device
	const uint32_t TPB = StaticThreadPerBlock1D;	
	const uint32_t shMemSize = rngMem.SharedMemorySize(TPB);
	const uint32_t totalWorkCount = pixelCount[0] * samplePerLocation *
									pixelCount[1] * samplePerLocation;
	// GPUSplits
	const auto splits = CudaSystem::GridStrideMultiGPUSplit(totalWorkCount, TPB, shMemSize,
															KCGenerateCameraRays<RayAuxData, AuxFunc>);

	for(int i = 0; i < static_cast<int>(CudaSystem::GPUList().size()); i++)
	{
		// Arguments
		const CudaGPU& gpu = CudaSystem::GPUList()[i];
		const int gpuId = gpu.DeviceId();
		// Generic Args
		const size_t localWorkCount = splits[i];
		const size_t localPixelCount1D = splits[i] / samplePerLocation / samplePerLocation;
		const Vector2i localPixel2D = Vector2i(localPixelCount1D % StaticThreadPerBlock2D_X,
											   localPixelCount1D / StaticThreadPerBlock2D_X);
				   
		// Kernel Specific Args
		// Output
		RayGMem* gRays = rayMem.RaysOut();
		RayAuxData* gAuxiliary = rayMem.RayAuxOut<RayAuxData>();
		// Input
		RNGGMem rngData = rngMem.RNGData(gpuId);

		Vector2i localPixelCount = Vector2i(localPixelCount1D % pixelCount[0],
											localPixelCount1D / pixelCount[0]);
		Vector2i localPixelStart = pixelStart + localPixelCount * i;
		Vector2i localPixelEnd = localPixelStart + localPixelCount;

		// Kernel Call
		CudaSystem::AsyncGridStrideKC_X
		(
			gpuId, shMemSize, localWorkCount,
			KCGenerateCameraRays<RayAuxData, AuxFunc>,
			// Args
			// Inputs
			gRays,
			gAuxiliary,
			// Input
			rngData,
			currentCam,
			samplePerLocation,
			resolution,
			localPixelStart,
			localPixelEnd,
			// Data to initialize auxiliary base data
			initialValues
		);
	}
	return currentRayCount;
}