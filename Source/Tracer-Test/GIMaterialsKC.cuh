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

    // Finalize
    Vector3f radiance = aux.irradianceFactor * gMatData.dAlbedo[matId];

    // Final point on a ray path
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
                                const BasicEstimatorData& estData,
                                // Input as global memory
                                const ConstantAlbedoMatData& gMatData,
                                const HitKey::Type& matId)
{
    assert(maxOutRay == 2);
    // Inputs
    const RayAuxBasic auxIn = aux;
    const RayReg rayIn = ray;
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
    auxOut.irradianceFactor = auxIn.irradianceFactor * nDotL * gMatData.dAlbedo[matId];

    // Russian Roulette
    float avgThroughput = auxOut.irradianceFactor.Dot(Vector3f(0.333f)) * 100.0f;
    //float avgThroughput = gMatData.dAlbedo[matId].Dot(Vector3f(0.333f)) * 100.0f;
    if(GPUEventEstimatorBasic::TerminatorFunc(auxOut.irradianceFactor, avgThroughput, rng))
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, 0);
        gBoundaryMat[0] = HitKey::InvalidKey;
        gOutRayAux[0] = {{99.0f, 99.0f, 99.0f}, 999, 999};
    }
    else
    {
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

    // Generate Light Ray
    float pdf;
    HitKey key;
    Vector3 lDirection;
    if(GPUEventEstimatorBasic::EstimatorFunc(key, lDirection, pdf,
                                             // Input
                                             auxOut.irradianceFactor,
                                             position,
                                             rng,
                                             //
                                             estData))
    {
        // We estimated a direction
        printf("NOONE HERE\n");
    }
    else
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, 1);
        gBoundaryMat[1] = HitKey::InvalidKey;
        gOutRayAux[1] = {{99.0f, 99.0f, 99.0f}, 999, 999};
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