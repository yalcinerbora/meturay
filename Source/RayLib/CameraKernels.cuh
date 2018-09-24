#pragma once

/**

Camera Ray Generation Kernel

*/

#include <cstdint>
#include <cuda_runtime.h>

#include "RayLib/Vector.h"
#include "RayLib/Camera.h"
#include "RayLib/RayStructs.h"

#include "RayLib/Random.cuh"

// Commands that initialize ray auxiliary data
template <class RayAuxGMem, class RayAuxBaseData>
using AuxInitFunc = void(*)(const RayAuxGMem gAux,
							const uint32_t writeLocation,
							// Input
							const RayAuxBaseData baseData,
							// Index
							const Vector2ui& globalPixelId,
							const Vector2ui& localSampleId,
							const uint32_t samplePerPixel);

// Templated Camera Ray Generation Kernel
template<class RayAuxGMem, class RayAuxBaseData,
		 AuxInitFunc<RayAuxGMem, RayAuxBaseData> AuxFunc>
__global__ void KCGenerateCameraRays(RayGMem* gRays,
									 RayAuxGMem gAuxiliary,
									 // Input
									 RNGGMem gRNGStates,
									 const CameraPerspective cam,
									 const uint32_t samplePerPixel,
									 const Vector2ui resolution,
									 const Vector2ui pixelStart,
									 const Vector2ui pixelCount,
									 // Data to initialize auxiliary base data
									 const RayAuxBaseData auxBaseData)
{
	extern __shared__ uint32_t sRandState[];
	RandomGPU rng(gRNGStates.state, sRandState);

	// Total work
	const uint32_t totalWorkCount = pixelCount[0] * pixelCount[1] * samplePerPixel * samplePerPixel;

	// Find world space window sizes
	float widthHalf = atanf(cam.fov[0] * 0.5f) * cam.nearPlane;
	float heightHalf = atanf(cam.fov[1] * 0.5f) * cam.nearPlane;

	// Camera Space pixel sizes
	Vector2 delta = Vector2((widthHalf * 2.0f) / static_cast<float>(resolution[0] * samplePerPixel),
							(heightHalf * 2.0f) / static_cast<float>(resolution[1] * samplePerPixel));

	// Camera Vector Correction
	Vector3 gaze = cam.gazePoint - cam.position;
	Vector3 right = Cross(gaze, cam.up).Normalize();
	Vector3 up = Cross(right, gaze).Normalize();
	gaze = Cross(up, right).Normalize();

	// Camera parameters
	Vector3 topLeft = cam.position
						- right *  widthHalf
						+ up * heightHalf
						+ gaze * cam.nearPlane;
	Vector3 pos = cam.position;

	// Kernel Grid-Stride Loop
	for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
		threadId < totalWorkCount;
		threadId += (blockDim.x * gridDim.x))
	{
		Vector2ui threadId2d = Vector2ui(threadId % (pixelCount[0] * samplePerPixel),
										 threadId / (pixelCount[0] * samplePerPixel));
		Vector2ui globalSampleId = (pixelStart * samplePerPixel) + threadId2d;
		Vector2ui globalPixelId = pixelStart + (threadId2d / samplePerPixel);
		Vector2ui localSampleId = pixelStart + (threadId2d % samplePerPixel);

		// Create random location over sample rectangle
		float dX = RandFloat01(rng);
		float dY = RandFloat01(rng);
		Vector2 randomOffset = Vector2(dX, dY);

		// Ray's world position over canvas
		Vector2 sampleDistance = Vector2(static_cast<float>(globalSampleId[0]),
										 static_cast<float>(globalSampleId[1])) * delta;
		Vector2 samplePointDistance = sampleDistance + randomOffset * delta;
		Vector3 samplePoint = topLeft + (samplePointDistance[0] * right) - (samplePointDistance[1] * up);
	
		// Generate Required Parameters
		Vector2ui localPixelId = globalPixelId - pixelStart;
		uint32_t pixelIdLinear = localPixelId[1] * pixelCount[0] + localPixelId[0];
		uint32_t sampleIdLinear = localSampleId[1] * samplePerPixel + localSampleId[0];
		Vector3 rayDir = (samplePoint - pos).Normalize();

		RayGMem ray =
		{
			pos,
			0,
			rayDir,
			FLT_MAX
		};

		// Initialize Auxiliary Data
		AuxFunc(gAuxiliary, auxBaseData,
				threadId,
				globalPixelId,
				localSampleId,
				samplePerPixel);
	}	
}