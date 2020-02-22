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

static constexpr uint32_t BASICPT_MAX_OUT_RAY = 2;

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
    if(aux.type == RayType::NEE_RAY || aux.depth == 1)
    {
        // Finalize
        Vector3f radiance = aux.radianceFactor * gMatData.dAlbedo[matId];

        // Final point on a ray path
        Vector4f output(radiance[0],
                        radiance[1],
                        radiance[2],
                        1.0f);
        ImageAccumulatePixel(gImage, aux.pixelId, output);
    }
}

__device__
inline void BasicDiffusePTShade(// Output
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
    assert(maxOutRay == BASICPT_MAX_OUT_RAY);
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
    if(auxIn.type == RayType::NEE_RAY)
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
    Vector3 direction = HemiDistribution::HemiCosineCDF(xi);
    //Vector3 direction = HemiDistribution::HemiUniformCDF(xi);
    direction.NormalizeSelf();

    // Direction vector is on surface space (hemisperical)
    // Convert it to normal oriented hemisphere
    QuatF q = QuatF::RotationBetweenZAxis(normal);
    direction = q.ApplyRotation(direction);

    // Cos Tetha
    float nDotL = max(normal.Dot(direction), 0.0f);
    //float nDotL = abs(normal.Dot(direction));

    float pdfMat = nDotL * MathConstants::InvPi;
    //float pdfMat = MathConstants::InvPi * 0.5f;

    // Illumination Calculation
    Vector3 reflectance = gMatData.dAlbedo[matId] * MathConstants::InvPi;
    auxOut0.radianceFactor = auxIn.radianceFactor * nDotL * reflectance / pdfMat;

    // Russian Roulette
    float avgThroughput = auxOut0.radianceFactor.Dot(Vector3f(0.333f)) * 100.0f;
    // float avgThroughput = gMatData.dAlbedo[matId].Dot(Vector3f(0.333f)) * 100.0f;

    if(auxIn.depth <= 3 &&
       !GPUEventEstimatorBasic::TerminatorFunc(auxOut0.radianceFactor,
                                               avgThroughput,
                                               rng))
    {
        // Advance slightly to prevent self intersection
        Vector3 pos = position + normal * MathConstants::Epsilon;

        // Write Ray
        rayOut.ray = RayF(direction, pos);
        rayOut.tMin = 0.0f;
        rayOut.tMax = INFINITY;
        // All done!
        // Write to global memory
        rayOut.Update(gOutRays, 0);
        gOutRayAux[0] = auxOut0;
        // We dont have any specific boundary mat for this
        // dont set material key
    }
    else
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, 0);
        gBoundaryMat[0] = HitKey::InvalidKey;
    }
    
    // Generate Light Ray
    float pdfLight;
    HitKey matLight;
    Vector3 lDirection;
    if(GPUEventEstimatorBasic::EstimatorFunc(matLight, lDirection, pdfLight,
                                             // Input
                                             auxOut0.radianceFactor,
                                             position,
                                             rng,
                                             //
                                             estData))
    {

        //printf("%X  ", matLight.value);

        // Advance slightly to prevent self intersection
        Vector3 pos = position + normal * MathConstants::Epsilon;
        // Write Ray
        rayOut.ray = RayF(lDirection, pos);
        rayOut.tMin = 0.0f;// MathConstants::Epsilon;
        rayOut.tMax = INFINITY;

        // Cos Tetha
        float nDotL = max(normal.Dot(lDirection), 0.0f);
        //float nDotL = abs(normal.Dot(direction));

        Vector3 lReflectance = gMatData.dAlbedo[matId] * MathConstants::InvPi;
        auxOut1.radianceFactor = auxIn.radianceFactor * nDotL * lReflectance / pdfLight;

        // All done!
        // Write to global memory
        rayOut.Update(gOutRays, 1);
        gOutRayAux[1] = auxOut1;
        gOutRayAux[1].type = RayType::NEE_RAY;
        gBoundaryMat[1] = matLight;
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

__device__
inline void BasicReflectPTShade(// Output
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
{}

__device__
inline void BasicRefractPTShade(// Output
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
                                const ConstantMediumMatData& gMatData,
                                const HitKey::Type& matId)
{}