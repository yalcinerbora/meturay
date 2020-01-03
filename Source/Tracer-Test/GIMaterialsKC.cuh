#pragma once

struct RayReg;
class RandomGPU;

#include "RayAuxStruct.h"
#include "MaterialDataStructs.h"
#include "SurfaceStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"

#include "TracerLib/ImageFunctions.cuh"
#include "TracerLib/EmptyEventEstimator.h"

#include <cuda_runtime.h>

using ConstantIrradianceMatData = ConstantAlbedoMatData;

__device__
inline void LightBoundaryShade(// Output
                               ImageGMem<Vector4f> gImage,
                               HitKey* gBoundaryMat,
                               //
                               RayGMem* gOutRays,
                               RayAuxBasic* gOutRayAux,
                               const uint32_t maxOutRay,
                               // Input as registers
                               const RayReg& ray,
                               const EmptySurface& surface,
                               const RayAuxBasic& aux,
                               // RNG
                               RandomGPU& rng,
                               // Event Estimator
                               const EmptyEstimatorData&,
                               // Input as global memory
                               const ConstantIrradianceMatData& gMatData,
                               const HitKey::Type& matId)
{
    assert(maxOutRay == 0);

    // Finalize
    Vector3f radiance = aux.accumFactor * gMatData.dAlbedo[matId];

    // Final point on a ray path
    //printf("Touched Light!\n");
    Vector4f output(radiance[0],
                    radiance[1],
                    radiance[2],
                    1.0f);
    ImageAccumulatePixel(gImage, aux.pixelId, output);
}

__device__
inline void BasicPathTraceShade(// Output
                                ImageGMem<Vector4f> gImage,
                                //
                                HitKey* gBoundaryMat,
                                RayGMem* gOutRays,
                                RayAuxBasic* gOutRayAux,
                                const uint32_t maxOutRay,
                                // Input as registers
                                const RayReg& ray,
                                const BasicSurface& surface,
                                const RayAuxBasic& aux,
                                // RNG
                                RandomGPU& rng,
                                // 
                                const BasicEstimatorData&,
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
    ////// Write color to pixel
    //Vector3f normalColor = (surface.normal + Vector3f(1.0f)) * 0.5f;
    //gImage[aux.pixelId][0] = normalColor[0];
    //gImage[aux.pixelId][1] = normalColor[1];
    //gImage[aux.pixelId][2] = normalColor[2];
    //return;

    assert(maxOutRay == 1);
    // Inputs
    const RayAuxBasic& auxIn = aux;
    const RayReg& rayIn = ray;
    // Outputs
    RayReg rayOut = {};
    RayAuxBasic auxOut = auxIn;    

    // Material calculation is done
    // continue to the determination of
    // ray direction over path

    // Ray Selection
    Vector3 position = rayIn.ray.AdvancedPos(rayIn.tMax);
    Vector3 normal = surface.normal;
    // Generate New Ray Directiion
    Vector2 xi(GPUDistribution::Uniform<float>(rng),
               GPUDistribution::Uniform<float>(rng));
    //Vector3 direction = CosineDist::HemiCosineCDF(xi);
    Vector3 direction = HemiDistribution::HemiUniformCDF(xi);
    direction.NormalizeSelf();
    // Direction vector is on surface space (hemisperical)
    // Convert it to normal oriented hemisphere
    QuatF q = QuatF::RotationBetweenZAxis(normal);
    direction = q.ApplyRotation(direction);

    //printf("%f, %f\n", xi[0], xi[1]);
    //printf("pos %f, %f, %f\n"
    //       "dir %f, %f, %f\n",
    //       position[0], position[1], position[2],
    //       direction[0], direction[1], direction[2]);

    // Cos Tetha
    float nDotL = max(normal.Dot(direction), 0.0f);

     // Illumination Calculation
    auxOut.accumFactor = auxIn.accumFactor * nDotL * gMatData.dAlbedo[matId];

    // TODO:
    //if material is emissive directly write current contribution

    // Russian Roulette
    //float maxFactor = fmax(auxOut.accumFactor[0],
    //                       fmax(auxOut.accumFactor[1],
    //                            auxOut.accumFactor[2]));
    float maxFactor = fmax(gMatData.dAlbedo[matId][0],
                           fmax(gMatData.dAlbedo[matId][1],
                                gMatData.dAlbedo[matId][2]));
    if(GPUDistribution::Uniform<float>(rng) >= maxFactor)
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = {};
        rDummy.ray = {Zero3, Zero3};
        rDummy.tMin = INFINITY;
        rDummy.tMax = INFINITY;

        // Gmem Write
        rDummy.Update(gOutRays, 0);
        gBoundaryMat[0] = HitKey::InvalidKey;
        return;
    }   
    auxOut.accumFactor *= 1.0f / maxFactor;

    
    //// Dummy ray to global memory
    //RayReg rDummy = {};
    //rDummy.ray = {Zero3, Zero3};
    //rDummy.tMin = INFINITY;
    //rDummy.tMax = INFINITY;
    //rDummy.Update(gOutRays, 0);
    //gBoundaryMat[0] = HitKey::InvalidKey;
    //// Write color to pixel
    //Vector3f color = (surface.normal + Vector3f(1.0f)) * 0.5f;
    ////Vector3f color = (direction + Vector3f(1.0f)) * 0.5f;
    //gImage[aux.pixelId][0] = color[0];
    //gImage[aux.pixelId][1] = color[1];
    //gImage[aux.pixelId][2] = color[2];
    //return;
    
    // Advance slightly to prevent self intersection
    position += normal * MathConstants::Epsilon;

    // Write Ray
    rayOut.ray = RayF(direction, position);
    rayOut.tMin = 0.001f;
    rayOut.tMax = INFINITY;

    // All done!
    // Write to global memory
    rayOut.Update(gOutRays, 0);
    gOutRayAux[0] = auxOut;
    // We dont have any specific boundary mat for this
    // dont set material key
}