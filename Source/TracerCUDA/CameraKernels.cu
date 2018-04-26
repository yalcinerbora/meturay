#include "CameraKernels.cuh"
#include "RayLib/Camera.h"
#include "RayLib/RayHitStructs.h"
#include "RayLib/Random.cuh"

__global__ void KCGenerateCameraRays(RayStackGMem gRays,
									 RandomStackGMem gRand,
									 const CameraPerspective cam,
									 const uint32_t samplePerPixel,
									 const Vector2ui resolution,
									 const Vector2ui pixelStart,
									 const Vector2ui pixelCount)
{
	extern __shared__ uint32_t sRandState[];
	RandomGPU rng(gRand.state, sRandState);
	
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
		float dX = 0.0f; //RandFloat01(rng);
		float dY = 0.0f; //RandFloat01(rng);
		Vector2 randomOffset = Vector2(dX, dY);

		// Ray's world position over canvas
		Vector2 sampleDistance = Vector2(static_cast<float>(globalSampleId[0]),
										 static_cast<float>(globalSampleId[1])) * delta;
		Vector3 sampleTopLeft = topLeft + (sampleDistance[0] * right)
										- (sampleDistance[1] * up);		
		Vector3 samplePoint = sampleTopLeft + randomOffset;
		

		// Generate Required Parameters
		uint32_t pixelIdLinear = globalPixelId[1] * resolution[0] + globalPixelId[0];
		uint32_t sampleIdLinear = localSampleId[1] * samplePerPixel + localSampleId[0];
		Vector3 rayDir = (samplePoint - pos).Normalize();

		// Write to GMem
		Vector4 posAndMed = Vector4(pos, 1.0f);
		Vec3AndUInt dirAndPix = {rayDir, pixelIdLinear};		
		Vec3AndUInt radAndSamp = {Zero3, sampleIdLinear};
		gRays.posAndMedium[threadId] = posAndMed;
		gRays.dirAndPixId[threadId] = dirAndPix;
		gRays.radAndSampId[threadId] = radAndSamp;
	}

	
}