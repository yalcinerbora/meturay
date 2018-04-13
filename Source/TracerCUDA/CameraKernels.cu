#include "CameraKernels.cuh"
//#include "RayLib/Camera.h"
//#include "RayLib/RayHitStructs.h"
//#include "RayLib/Random.cuh"

//__global__ void GenerateCameraRays(RayStackGMem gRays,
//								   RandomStackGMem gRand,
//								   const CameraPerspective cam, 
//								   const uint32_t samplePerPixel,
//								   const Vector2ui resolution,
//								   const Vector2ui pixelStart, 
//								   const Vector2ui pixelCount)
//{
//	////uint32_t samplePerPixelSqr = samplePerPixel * samplePerPixel;
//	////int32_t totalPixelCount = static_cast<int32_t>(samplePerPixelSqr * pixelCount[0] * pixelCount[1]);
//
//	//// Find world space window sizes
//	//float widthHalf = atanf(cam.fov[0] * 0.5f) * cam.near;
//	//float heightHalf = atanf(cam.fov[1] * 0.5f) * cam.near;
//
//	//// Camera Space pixel sizes
//	//Vector2 delta = Vector2((widthHalf * 2.0f) / static_cast<float>(resolution[0] * samplePerPixel),
//	//						(heightHalf * 2.0f) / static_cast<float>(resolution[1] * samplePerPixel));
//
//	//// Camera Vector Correction
//	//Vector3 gaze = cam.gazePoint - cam.position;
//	//Vector3 right = Cross(gaze, cam.up).Normalize();
//	//Vector3 up = Cross(right, gaze).Normalize();
//	//gaze = Cross(up, right);
//
//	//// Camera parameters
//	//Vector3 topLeft = cam.position
//	//					- right *  widthHalf
//	//					+ up * heightHalf
//	//					+ gaze * cam.near;
//	//Vector3 pos = cam.position;
//
//
//	//// Kernel Grid-Stride Loop
//	//for(Vector2ui threadId = Vector2ui(threadIdx.x + blockDim.x * blockIdx.x,
//	//								   threadIdx.y + blockDim.y * blockIdx.y);
//	//	threadId < pixelCount;
//	//	threadId += Vector2ui(blockDim.x, blockDim.y) * Vector2ui(gridDim.x, gridDim.y))
//	//{		
//	//	Vector2ui globalSampleId = pixelStart + threadId;
//	//	Vector2ui globalPixelId = pixelStart + (threadId / samplePerPixel);
//	//	Vector2ui localSampleId = pixelStart + (threadId % samplePerPixel);
//
//	//	// Create random location over sample rectangle
//	//	Vector2 randomOffset = Vector2();
//
//	//	// Ray's world position over canvas
//	//	Vector2 sampleDistance = Vector2(static_cast<float>(globalSampleId[0]),
//	//									 static_cast<float>(globalSampleId[1])) * delta;
//	//	Vector3 sampleTopLeft = topLeft + (sampleDistance[0] * right)
//	//									- (sampleDistance[1] * up);
//
//	//	// Add derivation
//	//	
//
//
//
//
//
//	//	//rays.dirAndPixId
//
//	//}
//
//	
//}