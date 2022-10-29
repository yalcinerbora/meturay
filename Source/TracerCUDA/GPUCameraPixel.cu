//#include "GPUCameraPixel.cuh"
//#include "CudaSystem.hpp"
//#include "RayLib/MemoryAlignment.h"
//
//__global__
//void KCConstructSingleGPUCameraPixel(GPUCameraPixel* gCameraLocations,
//                                     //
//                                     const GPUCameraI& baseCam,
//                                     Vector2i pixelIndex,
//                                     Vector2i resolution)
//{
//    if(threadIdx.x != 0) return;
//
//    GPUCameraPixel pCam = baseCam.GeneratePixelCamera(pixelIndex,
//                                                      resolution);
//
//    // TODO: WTF? this does not copy vptr
//    //*gCameraLocations = pCam;
//    // But this does?
//    new (gCameraLocations) GPUCameraPixel(pCam);
//}