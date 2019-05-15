#pragma once

struct RayReg;
class RandomGPU;

#include "RayAuxStruct.h"
#include "MaterialDataStructs.h"
#include "SurfaceStructs.h"

#include "RayLib/Constants.h"

__device__
inline void ConstantBoundaryMatShade(// Output
                                     Vector4* gImage,
                                     // Input as registers
                                     const RayReg& ray,
                                     const RayAuxBasic& aux,
                                     //
                                     RandomGPU& rng,
                                     // Input as global memory
                                     const ConstantBoundaryMatData& gMatData)
{
    Vector3f output = gMatData.backgroundColor * aux.totalRadiance;
    gImage[aux.pixelId][0] = output[0];
    gImage[aux.pixelId][1] = output[1];
    gImage[aux.pixelId][2] = output[2];
}

__device__
inline void BasicMatShade(// Output
                          Vector4f* gImage,
                          //
                          RayGMem* gOutRays,
                          RayAuxBasic* gOutRayAux,
                          const uint32_t maxOutRay,
                          // Input as registers
                          const RayReg& ray,
                          const EmptySurface& surface,
                          const RayAuxBasic& aux,
                          //
                          RandomGPU& rng,
                          // Input as global memory
                          const ConstantAlbedoMatData& gMatData,
                          const HitKey::Type& matId)
{
    gImage[aux.pixelId][0] = gMatData.dAlbedo[matId][0];
    gImage[aux.pixelId][1] = gMatData.dAlbedo[matId][1];
    gImage[aux.pixelId][2] = gMatData.dAlbedo[matId][2];
}

__device__
inline void BaryMatShade(// Output
                         Vector4f* gImage,
                         //
                         RayGMem* gOutRays,
                         RayAuxBasic* gOutRayAux,
                         const uint32_t maxOutRay,
                         // Input as registers
                         const RayReg& ray,
                         const BarySurface& surface,
                         const RayAuxBasic& aux,
                         //
                         RandomGPU& rng,
                         // Input as global memory
                         const NullData& gMatData,
                         const HitKey::Type& matId)
{
    gImage[aux.pixelId][0] = surface.baryCoords[0];
    gImage[aux.pixelId][1] = surface.baryCoords[1];
    gImage[aux.pixelId][2] = surface.baryCoords[2];
}

__device__
inline void SphrMatShade(// Output
                         Vector4f* gImage,
                         //
                         RayGMem* gOutRays,
                         RayAuxBasic* gOutRayAux,
                         const uint32_t maxOutRay,
                         // Input as registers
                         const RayReg& ray,
                         const SphrSurface& surface,
                         const RayAuxBasic& aux,
                         //
                         RandomGPU& rng,
                         // Input as global memory
                         const NullData& gMatData,
                         const HitKey::Type& matId)
{
    using namespace MathConstants;

    float piHalf = Pi * 0.5f;
    float twoPi = Pi * 0.5f;
    // Normalize spherical coords
    float tethaNorm = surface.sphrCoords[0] / twoPi;
    float phiNorm = (surface.sphrCoords[1] + piHalf) / Pi;

    gImage[aux.pixelId][0] = tethaNorm;
    gImage[aux.pixelId][1] = phiNorm;
    gImage[aux.pixelId][2] = 0.0f;
}