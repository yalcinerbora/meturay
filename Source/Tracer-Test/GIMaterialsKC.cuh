//#pragma once
//
//struct RayReg;
//class RandomGPU;
//
//#include "RayAuxStruct.h"
//#include "MaterialDataStructs.h"
//#include "SurfaceStructs.h"
//
//#include "RayLib/Constants.h"
//#include "RayLib/HemiDistribution.h"
//
//#include "TracerLib/ImageFunctions.cuh"
//
//#include <cuda_runtime.h>
//
//using IrradianceMatData = AlbedoMatData;
//
//static constexpr uint32_t BASICPT_MAX_OUT_RAY = 2;
//static constexpr uint32_t REFLECTPT_MAX_OUT_RAY = 1;
//static constexpr uint32_t REFRACTPT_MAX_OUT_RAY = 1;
//
//
//Vector3 DiffuseSample(// Sampled Output
//                      RayF& wo,
//                      float& pdf,
//                      // Input
//                      const Vector3& wi,
//                      const Vector3& pos,
//                      const EmptySurface& surface,
//                      // I-O
//                      RandomGPU& rng,
//                      // Constants
//                      const AlbedoMatData& matData,
//                      const HitKey::Type& matId)
//{
//    return matData.dAlbedo[matId];
//}
//
//__device__
//inline void LightBoundaryShade(// Output
//                               ImageGMem<Vector4f> gImage,
//                               HitKey* gBoundaryMat,
//                               //
//                               RayGMem* gOutRays,
//                               RayAuxBasic* gOutRayAux,
//                               const uint32_t maxOutRay,
//                               // Input as registers
//                               const RayReg& ray,
//                               const EmptySurface& surface,
//                               const RayAuxBasic& aux,
//                               // RNG
//                               RandomGPU& rng,
//                               // Event Estimator
//                               const BasicEstimatorData&,
//                               // Input as global memory
//                               const IrradianceMatData& gMatData,
//                               const HitKey::Type& matId)
//{
//    assert(maxOutRay == 0);
//
//    // Skip if light ray
//    if(aux.type == RayType::NEE_RAY || aux.type == RayType::CAMERA_RAY)
//    {
//        Vector3f radiance = aux.radianceFactor * gMatData.dAlbedo[matId];
//
//        // Final point on a ray path
//        Vector4f output(radiance[0],
//                        radiance[1],
//                        radiance[2],
//                        1.0f);
//        ImageAccumulatePixel(gImage, aux.pixelId, output);
//    }
//}
//
//__device__
//inline void BasicDiffusePTShade(// Output
//                                ImageGMem<Vector4f> gImage,
//                                //
//                                HitKey* gBoundaryMat,
//                                RayGMem* gOutRays,
//                                RayAuxBasic* gOutRayAux,
//                                const uint32_t maxOutRay,
//                                // Input as registers
//                                const RayReg& ray,
//                                const BasicSurface& surface,
//                                const RayAuxBasic& aux,
//                                // RNG
//                                RandomGPU& rng,
//                                // 
//                                const BasicEstimatorData& estData,
//                                // Input as global memory
//                                const AlbedoMatData& gMatData,
//                                const HitKey::Type& matId)
//{
//    assert(maxOutRay == BASICPT_MAX_OUT_RAY);
//    // Inputs
//    const RayAuxBasic& auxIn = aux;
//    const RayReg rayIn = ray;
//    // Outputs
//    RayReg rayOut = {};
//    RayAuxBasic auxOut0 = auxIn;
//    RayAuxBasic auxOut1 = auxIn;
//
//    auxOut0.depth++;
//    auxOut1.depth++;
//    auxOut0.type = RayType::PATH_RAY;
//    auxOut1.type = RayType::NEE_RAY;
//
//    // Skip if light ray
//    if(auxIn.type == RayType::NEE_RAY)
//    {
//        // Generate Dummy Ray and Terminate
//        RayReg rDummy = EMPTY_RAY_REGISTER;
//        rDummy.Update(gOutRays, 0);
//        rDummy.Update(gOutRays, 1);
//        gBoundaryMat[0] = HitKey::InvalidKey;
//        gBoundaryMat[1] = HitKey::InvalidKey;
//        return;
//    }
//
//    // Ray Selection
//    const Vector3 position = rayIn.ray.AdvancedPos(rayIn.tMax);
//    const Vector3 normal = surface.normal;
//    // Generate New Ray Directiion
//    Vector2 xi(GPUDistribution::Uniform<float>(rng),
//               GPUDistribution::Uniform<float>(rng));
//    Vector3 direction = HemiDistribution::HemiCosineCDF(xi);
//    direction.NormalizeSelf();
//
//    // Direction vector is on surface space (hemisperical)
//    // Convert it to normal oriented hemisphere (world space)
//    QuatF q = QuatF::RotationBetweenZAxis(normal);
//    direction = q.ApplyRotation(direction);
//
//    // Cos Tetha
//    float nDotL = max(normal.Dot(direction), 0.0f);
//    // Pdf
//    float pdfMat = nDotL * MathConstants::InvPi;
//
//    // Illumination Calculation
//    Vector3 reflectance = gMatData.dAlbedo[matId] * MathConstants::InvPi;
//    auxOut0.radianceFactor = auxIn.radianceFactor * nDotL * reflectance / pdfMat;
//
//    // Russian Roulette
//    float avgThroughput = auxOut0.radianceFactor.Dot(Vector3f(0.333f));
//    // float avgThroughput = gMatData.dAlbedo[matId].Dot(Vector3f(0.333f));
//
//    if(auxIn.depth <= 3 &&
//       !GPUEventEstimatorBasic::TerminatorFunc(auxOut0.radianceFactor,
//                                               avgThroughput,
//                                               rng))
//    {
//        // Advance slightly to prevent self intersection
//        Vector3 pos = position + normal * MathConstants::Epsilon;
//
//        // Write Ray
//        rayOut.ray = RayF(direction, pos);
//        rayOut.tMin = 0.0f;
//        rayOut.tMax = INFINITY;
//        // All done!
//        // Write to global memory
//        rayOut.Update(gOutRays, 0);
//        gOutRayAux[0] = auxOut0;
//        // We dont have any specific boundary mat for this
//        // dont set material key
//    }
//    else
//    {
//        // Generate Dummy Ray and Terminate
//        RayReg rDummy = EMPTY_RAY_REGISTER;
//        rDummy.Update(gOutRays, 0);
//        gBoundaryMat[0] = HitKey::InvalidKey;
//    }
//    
//    // Generate Light Ray
//    float pdfLight;
//    HitKey matLight;
//    Vector3 lDirection;
//    if(GPUEventEstimatorBasic::EstimatorFunc(matLight, lDirection, pdfLight,
//                                             // Input
//                                             auxOut0.radianceFactor,
//                                             position,
//                                             rng,
//                                             //
//                                             estData))
//    {
//        // Advance slightly to prevent self intersection
//        Vector3 pos = position + normal * MathConstants::Epsilon;
//        // Write Ray
//        rayOut.ray = RayF(lDirection, pos);
//        rayOut.tMin = 0.0f;// MathConstants::Epsilon;
//        rayOut.tMax = INFINITY;
//
//        // Cos Tetha
//        float nDotL = max(normal.Dot(lDirection), 0.0f);
//        //float nDotL = abs(normal.Dot(direction));
//
//        Vector3 lReflectance = gMatData.dAlbedo[matId] * MathConstants::InvPi;
//        auxOut1.radianceFactor = auxIn.radianceFactor * nDotL * lReflectance / pdfLight;
//
//        // All done!
//        // Write to global memory
//        rayOut.Update(gOutRays, 1);
//        gOutRayAux[1] = auxOut1;
//        gBoundaryMat[1] = matLight;
//    }
//    else
//    {
//        // Generate Dummy Ray and Terminate
//        RayReg rDummy = EMPTY_RAY_REGISTER;
//        rDummy.Update(gOutRays, 1);
//        gBoundaryMat[1] = HitKey::InvalidKey;
//    }
//}
//
//__device__
//inline void BasicReflectPTShade(// Output
//                                ImageGMem<Vector4f> gImage,
//                                //
//                                HitKey* gBoundaryMat,
//                                RayGMem* gOutRays,
//                                RayAuxBasic* gOutRayAux,
//                                const uint32_t maxOutRay,
//                                // Input as registers
//                                const RayReg& ray,
//                                const BasicSurface& surface,
//                                const RayAuxBasic& aux,
//                                // RNG
//                                RandomGPU& rng,
//                                // 
//                                const BasicEstimatorData& estData,
//                                // Input as global memory
//                                const ReflectMatData& gMatData,
//                                const HitKey::Type& matId)
//{
//    assert(maxOutRay == REFLECTPT_MAX_OUT_RAY);
//    // Inputs
//    const RayAuxBasic& auxIn = aux;
//    const RayReg rayIn = ray;
//    // Outputs
//    RayReg rayOut = EMPTY_RAY_REGISTER;
//    RayAuxBasic auxOut = auxIn;
//    auxOut.depth++;
//    // Do not change the ray type
//
//    // Skip if light ray
//    if(auxIn.type == RayType::NEE_RAY)
//    {
//        // Generate Dummy Ray and Terminate
//        rayOut.Update(gOutRays, 0);
//        gBoundaryMat[0] = HitKey::InvalidKey;
//        return;
//    }
//
//    // Fetch Mat
//    Vector4 matData = gMatData.dAlbedoAndRoughness[matId];
//    Vector3 albedo = matData;
//    float roughness = matData[3];
//
//    // Fetch Data
//    const Vector3 position = rayIn.ray.AdvancedPos(rayIn.tMax);
//    const Vector3 direction = rayIn.ray.getDirection();
//    const Vector3 normal = surface.normal;
//
//    // Russian Roulette
//    float avgThroughput = auxOut.radianceFactor.Dot(Vector3f(0.333f));
//    // float avgThroughput = gMatData.dAlbedo[matId].Dot(Vector3f(0.333f));
//
//    if(auxIn.depth <= 3 &&
//       !GPUEventEstimatorBasic::TerminatorFunc(auxOut.radianceFactor,
//                                               avgThroughput,
//                                               rng))
//    {
//         // Calculate Reflection   
//        if(roughness == 1.0f)
//        {
//            // Singularity Just Reflect
//            rayOut.ray = RayF(direction, position).Reflect(normal);
//            rayOut.ray.AdvanceSelf(MathConstants::Epsilon);
//            rayOut.tMin = 0.0f;
//            rayOut.tMax = INFINITY;
//
//            auxOut.radianceFactor *= albedo;
//            gOutRayAux[0] = auxOut;
//        }
//        else
//        {
//            // TODO:
//            // Use a normal distribution for sampling wrt. roughness
//        }
//    }
//
//    gBoundaryMat[0] = HitKey::InvalidKey;
//    rayOut.Update(gOutRays, 0);
//}
//
//__device__
//inline void BasicRefractPTShade(// Output
//                                ImageGMem<Vector4f> gImage,
//                                //
//                                HitKey* gBoundaryMat,
//                                RayGMem* gOutRays,
//                                RayAuxBasic* gOutRayAux,
//                                const uint32_t maxOutRay,
//                                // Input as registers
//                                const RayReg& ray,
//                                const BasicSurface& surface,
//                                const RayAuxBasic& aux,
//                                // RNG
//                                RandomGPU& rng,
//                                // 
//                                const BasicEstimatorData& estData,
//                                // Input as global memory
//                                const RefractMatData& gMatData,
//                                const HitKey::Type& matId)
//{
//    assert(maxOutRay == REFRACTPT_MAX_OUT_RAY);
//
//    // Inputs
//    const RayAuxBasic& auxIn = aux;
//    const RayReg rayIn = ray;
//    // Outputs
//    RayReg rayOut = EMPTY_RAY_REGISTER;
//    RayAuxBasic auxOut = auxIn;
//    auxOut.depth++;
//    // Do not change the ray type
//
//    // Skip if light ray
//    if(auxIn.type == RayType::NEE_RAY)
//    {
//        // Generate Dummy Ray and Terminate
//        rayOut.Update(gOutRays, 0);
//        gBoundaryMat[0] = HitKey::InvalidKey;
//        return;
//    }
//
//    // Fetch Mat
//    Vector4 matData = gMatData.dAlbedoAndIndex[matId];
//    Vector3 albedo = matData;
//    float index = matData[3];
//
//    // Fetch Data
//    const Vector3 position = rayIn.ray.AdvancedPos(rayIn.tMax);
//    const Vector3 direction = rayIn.ray.getDirection();
//    const Vector3 normal = surface.normal;
//
//    // Russian Roulette
//    float avgThroughput = auxOut.radianceFactor.Dot(Vector3f(0.333f));
//    // float avgThroughput = gMatData.dAlbedo[matId].Dot(Vector3f(0.333f));
//
//    if(auxIn.depth <= 3 &&
//       !GPUEventEstimatorBasic::TerminatorFunc(auxOut.radianceFactor,
//                                               avgThroughput,
//                                               rng))
//    {
//        // Check if we are exiting or entering
//        bool entering = (__half2float(auxIn.mediumIndex) == 1.0f);
//        float fromMedium = auxIn.mediumIndex;
//        float toMedium = (entering) ? index : 1.0f;
//        
//        // Utilize frenel term to select reflection or refraction
//        RayF r;
//        bool refracted = ray.ray.Refract(r, normal, fromMedium, toMedium);
//
//        if(entering)
//        {
//
//        }
//        else
//        {
//            // No frenel
//        }
//    }
//
//    gBoundaryMat[0] = HitKey::InvalidKey;
//    rayOut.Update(gOutRays, 0);
//}