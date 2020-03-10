#include "PathTracer.cuh"

#include "RayLib/TracerError.h"
//
//TracerVolume::TracerVolume(GPUBaseAcceleratorI& ba,
//                           AcceleratorGroupList&& ag,
//                           AcceleratorBatchMappings&& ab,
//                           MaterialGroupList&& mg,
//                           MaterialBatchMappings&& mb,
//                           GPUEventEstimatorI& ee,
//                           //
//                           const TracerParameters& parameters,
//                           uint32_t hitStructSize,
//                           const Vector2i maxMats,
//                           const Vector2i maxAccels,
//                           const HitKey baseBoundMatKey)
//    : TracerBaseLogic(ba,
//                      std::move(ag), std::move(ab),
//                      std::move(mg), std::move(mb),
//                      ee,
//                      parameters,
//                      hitStructSize,
//                      maxMats,
//                      maxAccels,
//                      baseBoundMatKey)
//{}
//
//TracerError TracerVolume::Initialize()
//{
//    return TracerError::OK;
//}
//
//uint32_t TracerVolume::GenerateRays(const CudaSystem& cudaSystem,
//                                    //
//                                    ImageMemory& imgMem,
//                                    RayMemory& rayMem, RNGMemory& rngMem,
//                                    const GPUSceneI& scene,
//                                    const CPUCamera& cam,
//                                    int samplePerLocation,
//                                    Vector2i resolution,
//                                    Vector2i pixelStart,
//                                    Vector2i pixelEnd)
//{
//    return 0;
//}