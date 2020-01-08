#pragma once

struct RayReg;
class RandomGPU;

#include "RayAuxStruct.h"
#include "MaterialDataStructs.h"
#include "SurfaceStructs.h"

#include "RayLib/Constants.h"
#include "RayLib/HemiDistribution.h"

#include "TracerLib/ImageFunctions.cuh"
#include "TracerLib/GPUEventEstimatorBasic.h"

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
                               const BasicEstimatorData&,
                               // Input as global memory
                               const ConstantIrradianceMatData& gMatData,
                               const HitKey::Type& matId)
{
    assert(maxOutRay == 0);

    // Skip if light ray
    if(aux.lightRay || aux.depth == 1)
    {
        // Finalize
        Vector3f radiance = aux.irradianceFactor * gMatData.dAlbedo[matId];

        // Final point on a ray path
        Vector4f output(radiance[0],
                        radiance[1],
                        radiance[2],
                        1.0f);
        ImageAccumulatePixel(gImage, aux.pixelId, output);
    }
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
                                const BasicEstimatorData& estData,
                                // Input as global memory
                                const ConstantAlbedoMatData& gMatData,
                                const HitKey::Type& matId)
{
    assert(maxOutRay == 2);
    // Inputs
    const RayAuxBasic& auxIn = aux;
    const RayReg rayIn = ray;
    // Outputs
    RayReg rayOut = {};
    RayAuxBasic auxOut0 = auxIn;
    RayAuxBasic auxOut1 = auxIn;

    auxOut0.depth++;
    auxOut1.depth++;

    // Skip if light ray
    if(auxIn.lightRay)
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, 0);
        rDummy.Update(gOutRays, 1);
        gBoundaryMat[0] = HitKey::InvalidKey;
        gBoundaryMat[1] = HitKey::InvalidKey;
        return;
    }



    // Ray Selection
    const Vector3 position = rayIn.ray.AdvancedPos(rayIn.tMax);
    const Vector3 normal = surface.normal;
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
    auxOut0.irradianceFactor = auxIn.irradianceFactor * nDotL * gMatData.dAlbedo[matId];

    // Russian Roulette
    float avgThroughput = auxOut0.irradianceFactor.Dot(Vector3f(0.333f)) * 100.0f;
    //float avgThroughput = gMatData.dAlbedo[matId].Dot(Vector3f(0.333f)) * 100.0f;
    if(GPUEventEstimatorBasic::TerminatorFunc(auxOut0.irradianceFactor, avgThroughput, rng))
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, 0);
        gBoundaryMat[0] = HitKey::InvalidKey;
    }
    else
    {
        // Advance slightly to prevent self intersection
        Vector3 pos = position + normal * MathConstants::Epsilon;

        // Write Ray
        rayOut.ray = RayF(direction, pos);
        rayOut.tMin = MathConstants::Epsilon;
        rayOut.tMax = INFINITY;
        // All done!
        // Write to global memory
        rayOut.Update(gOutRays, 0);
        gOutRayAux[0] = auxOut0;
        // We dont have any specific boundary mat for this
        // dont set material key
    }

    // Generate Light Ray
    float pdf;
    HitKey key;
    Vector3 lDirection;
    if(GPUEventEstimatorBasic::EstimatorFunc(key, lDirection, pdf,
                                             // Input
                                             auxOut0.irradianceFactor,
                                             position,
                                             rng,
                                             //
                                             estData))
    {
        // Advance slightly to prevent self intersection
        Vector3 pos = position + normal * MathConstants::Epsilon;        
        // Write Ray
        rayOut.ray = RayF(lDirection, pos);
        rayOut.tMin = MathConstants::Epsilon;
        rayOut.tMax = INFINITY;

        // Cos Tetha
        float nDotL = max(normal.Dot(lDirection), 0.0f);
        auxOut1.irradianceFactor = auxIn.irradianceFactor * nDotL * gMatData.dAlbedo[matId];

        // All done!
        // Write to global memory
        rayOut.Update(gOutRays, 1);
        gOutRayAux[1] = auxOut1;
        gOutRayAux[1].lightRay = true;
        // We dont have any specific boundary mat for this
        // dont set material key
    }
    else
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, 1);
        gBoundaryMat[1] = HitKey::InvalidKey;
    }

    
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
}