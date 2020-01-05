#pragma once

struct RayReg;
class RandomGPU;

#include "RayAuxStruct.h"
#include "MaterialDataStructs.h"
#include "SurfaceStructs.h"

#include "RayLib/Constants.h"

#include "TracerLib/ImageFunctions.cuh"
#include "TracerLib/GPUEventEstimatorEmpty.h"

__device__
inline void BasicMatShade(// Output
                          ImageGMem<Vector4f> gImage,
                          //
                          HitKey* gOutBoundMat,
                          RayGMem* gOutRays,
                          RayAuxBasic* gOutRayAux,
                          const uint32_t maxOutRay,
                          // Input as registers
                          const RayReg& ray,
                          const EmptySurface& surface,
                          const RayAuxBasic& aux,
                          //
                          RandomGPU& rng,
                          // Event Estimator
                          const EmptyEstimatorData&,
                          // Input as global memory
                          const ConstantAlbedoMatData& gMatData,
                          const HitKey::Type& matId)
{
    Vector4f output(gMatData.dAlbedo[matId][0],
                    gMatData.dAlbedo[matId][1],
                    gMatData.dAlbedo[matId][2],
                    1.0f);
    ImageAccumulatePixel(gImage, aux.pixelId, output);
}

__device__
inline void BaryMatShade(// Output
                         ImageGMem<Vector4f> gImage,
                         //
                         HitKey* gOutBoundMat,
                         RayGMem* gOutRays,
                         RayAuxBasic* gOutRayAux,
                         const uint32_t maxOutRay,
                         // Input as registers
                         const RayReg& ray,
                         const BarySurface& surface,
                         const RayAuxBasic& aux,
                         // RNG
                         RandomGPU& rng,
                         // Event Estimator
                         const EmptyEstimatorData&,
                         // Input as global memory
                         const NullData& gMatData,
                         const HitKey::Type& matId)
{
    Vector4f output(surface.baryCoords[0],
                    surface.baryCoords[1],
                    surface.baryCoords[2],
                    1.0f);
    ImageAccumulatePixel(gImage, aux.pixelId, output);
}

__device__
inline void SphrMatShade(// Output
                         ImageGMem<Vector4f> gImage,
                         //
                         HitKey* gOutBoundMat,
                         RayGMem* gOutRays,
                         RayAuxBasic* gOutRayAux,
                         const uint32_t maxOutRay,
                         // Input as registers
                         const RayReg& ray,
                         const SphrSurface& surface,
                         const RayAuxBasic& aux,
                         // RNG
                         RandomGPU& rng,
                         // Event Estimator
                         const EmptyEstimatorData&,
                         // Input as global memory
                         const NullData& gMatData,
                         const HitKey::Type& matId)
{
    Vector4f output(cos(surface.sphrCoords[0]),
                    sin(surface.sphrCoords[1]),
                    0.0f,
                    1.0f);
    ImageAccumulatePixel(gImage, aux.pixelId, output);
}