﻿#pragma once

#include "RayAuxStruct.cuh"

#include "GPULightI.h"
#include "GPUMediumI.h"
#include "GPUDirectLightSamplerI.h"
#include "PathNode.cuh"
#include "RayStructs.h"
#include "ImageStructs.h"

#include "TracerFunctions.cuh"
#include "TracerConstants.h"

#include "RayLib/RandomColor.h"

#include "STreeKC.cuh"
#include "DTreeKC.cuh"

struct PPGTracerGlobalState
{
    // Output Image
    ImageGMem<Vector4>              gImage;
    // Light Related
    const GPULightI**               gLightList;
    uint32_t                        totalLightCount;
    const GPUDirectLightSamplerI*   lightSampler;
    // Medium Related
    const GPUMediumI**              mediumList;
    uint32_t                        totalMediumCount;
    // SDTree Related
    const STreeGPU*                 gStree;
    const DTreeGPU**                gReadDTrees;
    DTreeGPU**                      gWriteDTrees;
    // Path Related
    PathGuidingNode*                gPathNodes;
    uint32_t                        maximumPathNodePerRay;
    // Options
    // Path Guiding
    bool                            rawPathGuiding;
    // Options for NEE
    bool                            nee;
    // Russian Roulette
    int                             rrStart;
};

struct PPGTracerLocalState
{
    bool    emptyPrimitive;
};

template <class EGroup>
__device__ __forceinline__
void PPGTracerBoundaryWork(// Output
                           HitKey* gOutBoundKeys,
                           RayGMem* gOutRays,
                           RayAuxPPG* gOutRayAux,
                           const uint32_t maxOutRay,
                           // Input as registers
                           const RayReg& ray,
                           const RayAuxPPG& aux,
                           const typename EGroup::Surface& surface,
                           const RayId rayId,
                           // I-O
                           PPGTracerLocalState& localState,
                           PPGTracerGlobalState& renderState,
                           RandomGPU& rng,
                           // Constants
                           const typename EGroup::GPUType& gLight)
{
    using GPUType = typename EGroup::GPUType;

    uint32_t pathStartIndex = aux.pathIndex * renderState.maximumPathNodePerRay;
    PathGuidingNode* gLocalPathNodes = renderState.gPathNodes + pathStartIndex;

    // Check Material Sample Strategy
    assert(maxOutRay == 0);

    const bool isCameraRay = aux.type == RayType::CAMERA_RAY;
    const bool isSpecularPathRay = aux.type == RayType::SPECULAR_PATH_RAY;
    // Always eval boundary mat if NEE is off
    // or NEE is on and hit endpoint and requested endpoint is same
    const GPULightI* requestedLight = renderState.gLightList[aux.endpointIndex];
    const bool isCorrectLight = (requestedLight->EndpointId() == gLight.EndpointId());
    const bool isCorrectNEERay = ((!renderState.nee) ||
                                  (isCorrectLight && aux.type == RayType::NEE_RAY));

    // Accumulate Light if
    if(isCorrectNEERay   || // We hit the correct light as a NEE ray
       isCameraRay       || // We hit as a camera ray which should not be culled when NEE is on
       isSpecularPathRay)   // We hit as spec ray which did not launched any NEE rays thus it should be hit
    {
        const RayF& r = ray.ray;
        Vector3 position = r.AdvancedPos(ray.tMax);
        const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);

        // Calculate Transmittance factor of the medium
        Vector3 transFactor = m.Transmittance(ray.tMax);
        Vector3 radianceFactor = aux.radianceFactor * transFactor;

        Vector3 emission = gLight.Emit(// Input
                                       -r.getDirection(),
                                       position,
                                       surface);

        // And accumulate pixel
        // and add as a sample
        Vector3f total = emission * radianceFactor;
        ImageAccumulatePixel(renderState.gImage,
                             aux.pixelIndex,
                             Vector4f(total, 1.0f));

        // Also backpropogate this radiance to the path nodes
        //if(emission != Vector3(0.0f))
        //    printf("AddingRadiance: E:(%f %f %f) RF:(%f %f %f) Path %u + %u\n",
        //           emission[0], emission[1], emission[2],
        //           radianceFactor[0], radianceFactor[1], radianceFactor[2],
        //           aux.pathIndex, aux.depth);
        if(total.HasNaN()) printf("NAN Found boundary!!!\n");
        // Accumulate to the write tree aswell (instead of creating a node for each NEE
        // end point directly accumulating to the tree will have less memory need)
        if(aux.type != RayType::CAMERA_RAY)
        {
            gLocalPathNodes[aux.depth].AccumRadianceDownChain(total, gLocalPathNodes);

            uint32_t dTreeIndex = gLocalPathNodes[aux.depth].nearestDTreeIndex;
            DTreeGPU* dWriteTree = renderState.gWriteDTrees[dTreeIndex];

            dWriteTree->AddRadianceToLeaf(r.getDirection(), Utility::RGBToLuminance(total),
                                          true);
            //dWriteTree->AddRadianceToLeaf(r.getDirection(), 1.0f);

            //if(total != Vector3f(0.0f))
            //{
            //    Vector3 position = r.AdvancedPos(ray.tMax);

            //    printf("Final Add - D: %f %f %f = %f\n",
            //           r.getDirection()[0], r.getDirection()[1], r.getDirection()[2],
            //           Utility::RGBToLuminance(total));
            //}
        }
    }
}

template <class MGroup>
__device__ __forceinline__
void PPGTracerPathWork(// Output
                       HitKey* gOutBoundKeys,
                       RayGMem* gOutRays,
                       RayAuxPPG* gOutRayAux,
                       const uint32_t maxOutRay,
                       // Input as registers
                       const RayReg& ray,
                       const RayAuxPPG& aux,
                       const typename MGroup::Surface& surface,
                       const RayId rayId,
                       // I-O
                       PPGTracerLocalState& localState,
                       PPGTracerGlobalState& renderState,
                       RandomGPU& rng,
                       // Constants
                       const typename MGroup::Data& gMatData,
                       const HitKey::Type matIndex)
{
    static constexpr Vector3 ZERO_3 = Zero3;

    // TODO: change this currently only first strategy is sampled
    static constexpr int PATH_RAY_INDEX = 0;
    static constexpr int NEE_RAY_INDEX = 1;
    //static constexpr int MIS_RAY_INDEX = 2;

    uint32_t pathStartIndex = aux.pathIndex * renderState.maximumPathNodePerRay;
    PathGuidingNode* gLocalPathNodes = renderState.gPathNodes + pathStartIndex;

    // Inputs
    // Current Ray
    const RayF& r = ray.ray;
    // Hit Position
    Vector3 position = r.AdvancedPos(ray.tMax);
    // Wi (direction is swapped as if it is coming out of the surface
    Vector3 wi = -(r.getDirection().Normalize());
    // Current ray's medium
    const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);

    // Check Material Sample Strategy
    uint32_t sampleCount = maxOutRay;
    // Check Material's specularity;
    float specularity = MGroup::Specularity(surface, gMatData, matIndex);
    bool isSpecularMat = (specularity >= TracerConstants::SPECULAR_TRESHOLD);

    // Invalid Ray Write Helper Function
    auto InvalidRayWrite = [&gOutRays, &gOutBoundKeys, &gOutRayAux, &sampleCount](int index)
    {
        assert(index < sampleCount);

        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, index);
        gOutBoundKeys[index] = HitKey::InvalidKey;
        gOutRayAux[index].pixelIndex = UINT32_MAX;
    };

    // If NEE ray hits to this material
    // just skip since this is not a light material
    if(aux.type == RayType::NEE_RAY)
    {
        // Write invalids for out rays
        for(uint32_t i = 0; i < sampleCount; i++)
            InvalidRayWrite(i);
        return;
    }

    // Calculate Transmittance factor of the medium
    Vector3 transFactor = m.Transmittance(ray.tMax);
    Vector3 radianceFactor = aux.radianceFactor * transFactor;

    // Sample the emission if avail
    if(MGroup::IsEmissive(gMatData, matIndex))
    {
        Vector3 emission = MGroup::Emit(// Input
                                        wi,
                                        position,
                                        m,
                                        //
                                        surface,
                                        // Constants
                                        gMatData,
                                        matIndex);

        Vector3f total = emission * radianceFactor;
        // Output image
        ImageAccumulatePixel(renderState.gImage,
                             aux.pixelIndex,
                             Vector4f(total, 1.0f));

        //if(emission != Vector3(0.0f))
        //    printf("AddingRadiance: E:(%f %f %f) RF:(%f %f %f) Path %u + %u\n",
        //           emission[0], emission[1], emission[2],
        //           radianceFactor[0], radianceFactor[1], radianceFactor[2],
        //           aux.pathIndex, aux.depth);

        // Accumulate this to the paths aswell
        if(total.HasNaN()) printf("NAN Found emissive!!!\n");
        gLocalPathNodes[aux.depth].AccumRadianceDownChain(total, gLocalPathNodes);
    }

    // If this material does not require to have any samples just quit
    // no need to sat any ray invalid since there wont be any allocated rays
    if(sampleCount == 0) return;

    // Find nearest DTree
    uint32_t dTreeIndex = 0;
    renderState.gStree->AcquireNearestDTree(dTreeIndex, position);

    float pdf;
    RayF rayPath;
    Vector3f reflectance;
    const GPUMediumI* outM = &m;
    //if(!isSpecularMat)
    //{
    //    // Sample a path using SDTree
    //    const DTreeGPU* dReadTree = renderState.gReadDTrees[dTreeIndex];
    //    Vector3f direction = dReadTree->Sample(pdf, rng);

    //    //if(isnan(pdf) | direction.HasNaN())
    //    //    printf("pdf % f, dir % f, % f, % f\n", pdf,
    //    //           direction[0], direction[1], direction[2]);
    //    //
    //    //printf("D: %f, %f, %f\n", direction[0], direction[1], direction[2]);
    //    //printf("%u  ", dTreeIndex);
    //    //
    //    //reflectance = Utility::RandomColorRGB(dTreeIndex);
    //    // Calculate BxDF
    //    reflectance = MGroup::Evaluate(// Input
    //                                   direction,
    //                                   wi,
    //                                   position,
    //                                   m,
    //                                   //
    //                                   surface,
    //                                   // Constants
    //                                   gMatData,
    //                                   matIndex);

    //    if(reflectance.HasNaN())
    //        printf("NAN REFL %f %f %f\n",
    //               reflectance[0], reflectance[1], reflectance[2]);

    //    rayPath = RayF(direction, position);
    //    rayPath.AdvanceSelf(MathConstants::Epsilon);
    //}
    //else
    {
        // Sample the BxDF
        reflectance = MGroup::Sample(// Outputs
                                     rayPath, pdf, outM,
                                     // Inputs
                                     wi,
                                     position,
                                     m,
                                     //
                                     surface,
                                     // I-O
                                     rng,
                                     // Constants
                                     gMatData,
                                     matIndex,
                                     0);
    }


    // Factor the radiance of the surface
    Vector3f pathRadianceFactor = radianceFactor * reflectance;
    // Check singularities
    pathRadianceFactor = (pdf == 0.0f) ? Zero3 : (pathRadianceFactor / pdf);

    if(pathRadianceFactor.HasNaN())
        printf("NAN PATH R: %f %f %f = {%f %f %f} * {%f %f %f} / %f  \n",
               pathRadianceFactor[0], pathRadianceFactor[1], pathRadianceFactor[2],
               radianceFactor[0], radianceFactor[1], radianceFactor[2],
               reflectance[0], reflectance[1], reflectance[2], pdf);

    // Check Russian Roulette
    float avgThroughput = pathRadianceFactor.Dot(Vector3f(0.333f));
    bool terminateRay = ((aux.depth > renderState.rrStart) &&
                         TracerFunctions::RussianRoulette(pathRadianceFactor, avgThroughput, rng));

    // Do not terminate rays ever for specular mats
    if((!terminateRay || isSpecularMat) &&
        // Do not waste rays on zero radiance paths
       pathRadianceFactor != ZERO_3)
    {
        // Write Ray
        RayReg rayOut;
        rayOut.ray = rayPath;
        rayOut.tMin = 0.0f;
        rayOut.tMax = INFINITY;
        rayOut.Update(gOutRays, PATH_RAY_INDEX);

        // Write Aux
        RayAuxPPG auxOut = aux;
        auxOut.mediumIndex = static_cast<uint16_t>(outM->GlobalIndex());
        auxOut.radianceFactor = pathRadianceFactor;
        auxOut.type = (isSpecularMat) ? RayType::SPECULAR_PATH_RAY : RayType::PATH_RAY;
        auxOut.depth++;
        gOutRayAux[PATH_RAY_INDEX] = auxOut;
    }
    else InvalidRayWrite(PATH_RAY_INDEX);


    // Record this intersection on path chain
    uint8_t currentDepth = aux.depth + 1;
    uint8_t prevDepth = aux.depth;
    PathGuidingNode node;
    //printf("WritingNode PC:(%u %u) W:(%f, %f, %f) RF:(%f, %f, %f) Path: %u DT %u\n",
    //       static_cast<uint32_t>(prevDepth), static_cast<uint32_t>(currentDepth),
    //       position[0], position[1], position[2],
    //       pathRadianceFactor[0], pathRadianceFactor[1], pathRadianceFactor[2],
    //       aux.pathIndex, dTreeIndex);

    node.prevNext[1] = PathGuidingNode::InvalidIndex;
    node.prevNext[0] = prevDepth;
    node.worldPosition = position;
    node.nearestDTreeIndex = dTreeIndex;
    node.radFactor = pathRadianceFactor;
    node.totalRadiance = Zero3;
    gLocalPathNodes[currentDepth] = node;
    // Set Previous Path node's next index
    gLocalPathNodes[prevDepth].prevNext[1] = currentDepth;

    // Dont launch NEE if not requested
    // or material is highly specular
    if(!renderState.nee) return;

    //============================//
    //        NEE PORTION         //
    //============================//
    // Renderer requested a NEE Ray but material is highly specular
    // Check if nee is requested
    if(isSpecularMat && maxOutRay == 1)
        return;
    else if(isSpecularMat)
    {
        // Write invalid rays then return
        InvalidRayWrite(NEE_RAY_INDEX);
        //if(renderState.directLightMIS)
        //    InvalidRayWrite(MIS_RAY_INDEX);
        return;
    }

    // Material is not specular & tracer requested a NEE ray
    // Generate a NEE Ray
    float pdfLight, lDistance;
    HitKey matLight;
    Vector3 lDirection;
    uint32_t lightIndex;
    Vector3f neeReflectance = Zero3;

    if(renderState.lightSampler->SampleLight(matLight,
                                             lightIndex,
                                             lDirection,
                                             lDistance,
                                             pdfLight,
                                             // Input
                                             position,
                                             rng))
    {
        // Evaluate mat for this direction
        neeReflectance = MGroup::Evaluate(// Input
                                          lDirection,
                                          wi,
                                          position,
                                          m,
                                          //
                                          surface,
                                          // Constants
                                          gMatData,
                                          matIndex);
    }

    // Check if mis ray should be sampled
    //bool launchedMISRay = (renderState.directLightMIS &&
    //                       // Check if light can be sampled (meaning it is not a
    //                       // dirac delta light (point light spot light etc.)
    //                       renderState.lightList[lightIndex]->CanBeSampled());
    bool launchedMISRay = false;

    float pdfNEE = pdfLight;
    if(launchedMISRay)
    {
        float pdfBxDF = MGroup::Pdf(lDirection,
                                    wi,
                                    position,
                                    m,
                                    //
                                    surface,
                                    gMatData,
                                    matIndex);

        pdfNEE /= TracerFunctions::PowerHeuristic(1, pdfLight, 1, pdfBxDF);

        // PDF can become NaN if both BxDF pdf and light pdf is both zero
        // (meaning both sampling schemes does not cover this direction)
        if(isnan(pdfNEE)) pdfNEE = 0.0f;
    }

    // Do not waste a ray if material does not reflect
    // towards light's sampled position
    Vector3 neeRadianceFactor = radianceFactor * neeReflectance;
    neeRadianceFactor = (pdfNEE == 0.0f) ? Zero3 : (neeRadianceFactor / pdfNEE);
    if(neeRadianceFactor != ZERO_3)
    {
        // Generate & Write Ray
        RayF rayNEE = RayF(lDirection, position);
        rayNEE.AdvanceSelf(MathConstants::Epsilon);

        RayReg rayOut;
        rayOut.ray = rayNEE;
        rayOut.tMin = 0.0f;
        rayOut.tMax = lDistance;
        rayOut.Update(gOutRays, NEE_RAY_INDEX);

        RayAuxPPG auxOut = aux;
        auxOut.radianceFactor = neeRadianceFactor;
        auxOut.endpointIndex = lightIndex;
        auxOut.type = RayType::NEE_RAY;
        auxOut.depth++;

        gOutRayAux[NEE_RAY_INDEX] = auxOut;
        gOutBoundKeys[NEE_RAY_INDEX] = matLight;
    }
    else InvalidRayWrite(NEE_RAY_INDEX);

    //// Check MIS Ray return if not requested (since no ray is allocated for it)
    //if(!renderState.directLightMIS) return;

    //// Sample Another ray for MIS (from BxDF)
    //float pdfMIS = 0.0f;
    //RayF rayMIS; const GPUMediumI* outMMIS;
    //Vector3f misReflectance = Zero3;
    //if(launchedMISRay)
    //{
    //    misReflectance = MGroup::Sample(// Outputs
    //                                    rayMIS, pdfMIS, outMMIS,
    //                                    // Inputs
    //                                    wi,
    //                                    position,
    //                                    m,
    //                                    //
    //                                    surface,
    //                                    // I-O
    //                                    rng,
    //                                    // Constants
    //                                    gMatData,
    //                                    matIndex,
    //                                    0);

    //    // Find out the pdf of the light
    //    float pdfLightM, pdfLightC;
    //    renderState.lightSampler->Pdf(pdfLightM, pdfLightC,
    //                                   lightIndex, position,
    //                                   rayMIS.getDirection());
    //    // We are subsampling (discretely sampling) a single light
    //    // pdf of BxDF should also incorporate this
    //    pdfMIS *= pdfLightM;

    //    pdfMIS /= TracerFunctions::PowerHeuristic(1, pdfMIS, 1, pdfLightC * pdfLightM);

    //    // PDF can become NaN if both BxDF pdf and light pdf is both zero
    //    // (meaning both sampling schemes does not cover this direction)
    //    if(isnan(pdfMIS)) pdfMIS = 0.0f;
    //}

    //// Calculate Combined PDF
    //Vector3 misRadianceFactor = radianceFactor * misReflectance;
    //misRadianceFactor = (pdfMIS == 0.0f) ? Zero3 : (misRadianceFactor / pdfMIS);
    //if(launchedMISRay && misRadianceFactor != ZERO_3)
    //{
    //    // Write Ray
    //    RayReg rayOut;
    //    rayOut.ray = rayMIS;
    //    rayOut.tMin = 0.0f;
    //    rayOut.tMax = INFINITY;
    //    rayOut.Update(gOutRays, MIS_RAY_INDEX);

    //    // Write Aux
    //    RayAuxPath auxOut = aux;
    //    auxOut.mediumIndex = static_cast<uint16_t>(outMMIS->GlobalIndex());
    //    auxOut.radianceFactor = misRadianceFactor;
    //    auxOut.endPointIndex = lightIndex;
    //    auxOut.type = RayType::NEE_RAY;

    //    gOutBoundKeys[MIS_RAY_INDEX] = matLight;
    //    gOutRayAux[MIS_RAY_INDEX] = auxOut;
    //}
    //else InvalidRayWrite(MIS_RAY_INDEX);

    //// All Done!
    //// Dont launch NEE if not requested
    //// or material is highly specula
    //if(!renderState.nee) return;

    //// Renderer requested a NEE Ray but material is highly specular
    //// Check if nee is requested
    //if(isSpecularMat && maxOutRay == 1)
    //    return;
    //else if(isSpecularMat)
    //{
    //    // Write invalid rays then return
    //    InvalidRayWrite(NEE_RAY_INDEX);
    //    if(renderState.directLightMIS)
    //        InvalidRayWrite(MIS_RAY_INDEX);
    //    return;
    //}

    //// Material is not specular & tracer requested a NEE ray
    //// Generate a NEE Ray
    //float pdfLight, lDistance;
    //HitKey matLight;
    //Vector3 lDirection;
    //uint32_t lightIndex;
    //Vector3f neeReflectance = Zero3;
    //if(renderState.lightSampler->SampleLight(matLight,
    //                                          lightIndex,
    //                                          lDirection,
    //                                          lDistance,
    //                                          pdfLight,
    //                                          // Input
    //                                          position,
    //                                          rng))
    //{
    //    // Evaluate mat for this direction
    //    neeReflectance = MGroup::Evaluate(// Input
    //                                      lDirection,
    //                                      wi,
    //                                      position,
    //                                      m,
    //                                      //
    //                                      surface,
    //                                      // Constants
    //                                      gMatData,
    //                                      matIndex);
    //}

    //// Check if mis ray should be sampled
    //bool launchedMISRay = (renderState.directLightMIS &&
    //                       // Check if light can be sampled (meaning it is not a
    //                       // dirac delta light (point light spot light etc.)
    //                       renderState.lightList[lightIndex]->CanBeSampled());

    //float pdfNEE = pdfLight;
    //if(launchedMISRay)
    //{
    //    float pdfBxDF = MGroup::Pdf(lDirection,
    //                                wi,
    //                                position,
    //                                m,
    //                                //
    //                                surface,
    //                                gMatData,
    //                                matIndex);

    //    pdfNEE /= TracerFunctions::PowerHeuristic(1, pdfLight, 1, pdfBxDF);

    //    // PDF can become NaN if both BxDF pdf and light pdf is both zero
    //    // (meaning both sampling schemes does not cover this direction)
    //    if(isnan(pdfNEE)) pdfNEE = 0.0f;
    //}

    //// Do not waste a ray if material does not reflect
    //// towards light's sampled position
    //Vector3 neeRadianceFactor = radianceFactor * neeReflectance;
    //neeRadianceFactor = (pdfNEE == 0.0f) ? Zero3 : (neeRadianceFactor / pdfNEE);
    //if(neeRadianceFactor != ZERO_3)
    //{
    //    // Generate & Write Ray
    //    RayF rayNEE = RayF(lDirection, position);
    //    rayNEE.AdvanceSelf(MathConstants::Epsilon);

    //    RayReg rayOut;
    //    rayOut.ray = rayNEE;
    //    rayOut.tMin = 0.0f;
    //    rayOut.tMax = lDistance;
    //    rayOut.Update(gOutRays, NEE_RAY_INDEX);

    //    RayAuxPath auxOut = aux;
    //    auxOut.radianceFactor = neeRadianceFactor;
    //    auxOut.endPointIndex = lightIndex;
    //    auxOut.type = RayType::NEE_RAY;

    //    gOutRayAux[NEE_RAY_INDEX] = auxOut;
    //    gOutBoundKeys[NEE_RAY_INDEX] = matLight;
    //}
    //else InvalidRayWrite(NEE_RAY_INDEX);

    //// Check MIS Ray return if not requested (since no ray is allocated for it)
    //if(!renderState.directLightMIS) return;

    //// Sample Another ray for MIS (from BxDF)
    //float pdfMIS = 0.0f;
    //RayF rayMIS; const GPUMediumI* outMMIS;
    //Vector3f misReflectance = Zero3;
    //if(launchedMISRay)
    //{
    //    misReflectance = MGroup::Sample(// Outputs
    //                                    rayMIS, pdfMIS, outMMIS,
    //                                    // Inputs
    //                                    wi,
    //                                    position,
    //                                    m,
    //                                    //
    //                                    surface,
    //                                    // I-O
    //                                    rng,
    //                                    // Constants
    //                                    gMatData,
    //                                    matIndex,
    //                                    0);

    //    // Find out the pdf of the light
    //    float pdfLightM, pdfLightC;
    //    renderState.lightSampler->Pdf(pdfLightM, pdfLightC,
    //                                   lightIndex, position,
    //                                   rayMIS.getDirection());
    //    // We are subsampling (discretely sampling) a single light
    //    // pdf of BxDF should also incorporate this
    //    pdfMIS *= pdfLightM;

    //    pdfMIS /= TracerFunctions::PowerHeuristic(1, pdfMIS, 1, pdfLightC * pdfLightM);

    //    // PDF can become NaN if both BxDF pdf and light pdf is both zero
    //    // (meaning both sampling schemes does not cover this direction)
    //    if(isnan(pdfMIS)) pdfMIS = 0.0f;
    //}

    //// Calculate Combined PDF
    //Vector3 misRadianceFactor = radianceFactor * misReflectance;
    //misRadianceFactor = (pdfMIS == 0.0f) ? Zero3 : (misRadianceFactor / pdfMIS);
    //if(launchedMISRay && misRadianceFactor != ZERO_3)
    //{
    //    // Write Ray
    //    RayReg rayOut;
    //    rayOut.ray = rayMIS;
    //    rayOut.tMin = 0.0f;
    //    rayOut.tMax = INFINITY;
    //    rayOut.Update(gOutRays, MIS_RAY_INDEX);

    //    // Write Aux
    //    RayAuxPath auxOut = aux;
    //    auxOut.mediumIndex = static_cast<uint16_t>(outMMIS->GlobalIndex());
    //    auxOut.radianceFactor = misRadianceFactor;
    //    auxOut.endPointIndex = lightIndex;
    //    auxOut.type = RayType::NEE_RAY;

    //    gOutBoundKeys[MIS_RAY_INDEX] = matLight;
    //    gOutRayAux[MIS_RAY_INDEX] = auxOut;
    //}
    //else InvalidRayWrite(MIS_RAY_INDEX);

    //// All Done!
}