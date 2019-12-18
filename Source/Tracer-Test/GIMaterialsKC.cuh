#pragma once

struct RayReg;
class RandomGPU;

#include "RayAuxStruct.h"
#include "MaterialDataStructs.h"
#include "SurfaceStructs.h"

#include "RayLib/Constants.h"

using ConstantIrradianceMatData = ConstantAlbedoMatData;

__device__
inline void LightBoundaryShade(// Output
                               Vector4f* gImage,
                               HitKey* gBoundaryMat,
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
                               const ConstantIrradianceMatData& gMatData,
                               const HitKey::Type& matId)
{
    assert(maxOutRay == 0);
    // Final point on a ray path
    // TODO:
    gImage[aux.pixelId][0] = gMatData.dAlbedo[matId][0];
    gImage[aux.pixelId][1] = gMatData.dAlbedo[matId][1];
    gImage[aux.pixelId][2] = gMatData.dAlbedo[matId][2];
    return;

}

__device__
inline void BasicPathTraceShade(// Output
                                Vector4f* gImage,
                                //
                                HitKey* gBoundaryMat,
                                RayGMem* gOutRays,
                                RayAuxBasic* gOutRayAux,
                                const uint32_t maxOutRay,
                                // Input as registers
                                const RayReg& ray,
                                const BasicSurface& surface,
                                const RayAuxBasic& aux,
                                //
                                RandomGPU& rng,
                                // Input as global memory
                                const ConstantAlbedoMatData& gMatData,
                                const HitKey::Type& matId)
{
    //// Dummy ray to global memory
    //RayReg rDummy = {};
    //rDummy.ray = {Zero3, Zero3};
    //rDummy.tMin = INFINITY;
    //rDummy.tMax = INFINITY;
    //rDummy.Update(gOutRays, 0);
    //gBoundaryMat[0] = HitKey::InvalidKey;

    //printf("matId %d\n", matId);

    // Write color to pixel
    gImage[aux.pixelId][0] = gMatData.dAlbedo[matId][0];
    gImage[aux.pixelId][1] = gMatData.dAlbedo[matId][1];
    gImage[aux.pixelId][2] = gMatData.dAlbedo[matId][2];
    return;

    assert(maxOutRay == 1);
    // Inputs
    RayAuxBasic auxIn = aux;
    RayReg rayIn = ray;
    // Outputs
    RayReg rayOut = {};
    RayAuxBasic auxOut = {};

    // Illumination Calculation
    Vector3 rad = auxIn.totalRadiance;
    auxOut.totalRadiance = rad * gMatData.dAlbedo[matId];
    // Material calculation is done
    // continue to the determination of
    // ray direction over path

    // Ray Selection
    Vector3 position = rayIn.ray.AdvancedPos(rayIn.tMax);
    Vector3 normal = surface.normal;
    // Generate New Ray Directiion
    Vector2 xi(GPURand::ZeroOne<float>(rng),
               GPURand::ZeroOne<float>(rng));
    Vector3 direction = CosineDist::HemiICDF(xi);

    // Direction vector is on surface space (hemisperical)
    // Convert it to normal oriented hemisphere
    QuatF q = QuatF::RotationBetweenZAxis(normal);
    direction = q.ApplyRotation(direction);

    // Advance slightly to prevent self intersection
    position += direction * MathConstants::Epsilon;

    // Write Ray
    rayOut.ray = {direction, position};
    rayOut.tMin = 0.001f;
    rayOut.tMax = INFINITY;

    // All done!
    // Write to global memory
    rayOut.Update(gOutRays, 0);
    gOutRayAux[0] = auxOut;
}