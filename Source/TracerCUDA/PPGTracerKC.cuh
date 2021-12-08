#pragma once

#include "RayAuxStruct.cuh"

#include "GPULightI.h"
#include "GPUMediumI.h"
#include "GPUDirectLightSamplerI.h"
#include "PathNode.cuh"
#include "RayStructs.h"
#include "ImageStructs.h"
#include "WorkOutputWriter.cuh"

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
    const GPUDirectLightSamplerI*   gLightSampler;
    // Medium Related
    const GPUMediumI**              mediumList;
    uint32_t                        totalMediumCount;
    // SDTree Related
    const STreeGPU*                 gStree;
    const DTreeGPU*                 gReadDTrees;
    DTreeGPU*                       gWriteDTrees;
    // Path Related
    PathGuidingNode*                gPathNodes;
    uint32_t                        maximumPathNodePerRay;
    // Options
    // Path Guiding
    bool                            rawPathGuiding;
    // Options for NEE
    bool                            directLightMIS;
    bool                            nee;
    // Russian Roulette
    int                             rrStart;
};

struct PPGTracerLocalState
{
    bool    emptyPrimitive;
};

__device__ __forceinline__
uint8_t DeterminePathIndex(uint8_t depth)
{
    return depth - 1;
}

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

    // Current Path
    const uint32_t pathStartIndex = aux.pathIndex * renderState.maximumPathNodePerRay;
    PathGuidingNode* gLocalPathNodes = renderState.gPathNodes + pathStartIndex;

    // Check Material Sample Strategy
    assert(maxOutRay == 0);

    const bool isPathRayAsMISRay = renderState.directLightMIS && (aux.type == RayType::PATH_RAY);
    const bool isCameraRay = aux.type == RayType::CAMERA_RAY;
    const bool isSpecularPathRay = aux.type == RayType::SPECULAR_PATH_RAY;
    const bool isNeeRayNEEOn = renderState.nee && aux.type == RayType::NEE_RAY;
    const bool isPathRayNEEOff = (!renderState.nee) && (aux.type == RayType::PATH_RAY ||
                                                        aux.type == RayType::SPECULAR_PATH_RAY);
    // Always eval boundary mat if NEE is off
    // or NEE is on and hit endpoint and requested endpoint is same
    const GPULightI* requestedLight = (isNeeRayNEEOn) ? renderState.gLightList[aux.endpointIndex] : nullptr;
    const bool isCorrectNEERay = (isNeeRayNEEOn && (requestedLight->EndpointId() == gLight.EndpointId()));

    float misWeight = 1.0f;
    if(isPathRayAsMISRay)
    {
        Vector3 position = ray.ray.AdvancedPos(ray.tMax);
        Vector3 direction = ray.ray.getDirection().Normalize();

        // Find out the pdf of the light
        float pdfLightM, pdfLightC;
        renderState.gLightSampler->Pdf(pdfLightM, pdfLightC,
                                       //
                                       gLight.GlobalLightIndex(),
                                       ray.tMax,
                                       position,
                                       direction,
                                       surface.worldToTangent);

        // We are sub-sampling (discretely sampling) a single light
        // pdf of BxDF should also incorporate this
        float bxdfPDF = aux.prevPDF;
        misWeight = TracerFunctions::PowerHeuristic(1, bxdfPDF,
                                                    1, pdfLightC * pdfLightM);
    }

    // Calculate the total contribution
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

    // And accumulate pixel// and add as a sample
    Vector3f total = emission * radianceFactor;
    // Incorporate MIS weight if applicable
    // if path ray hits a light misWeight is calculated
    // else misWeight is 1.0f
    total *= misWeight;

    // Previous Path's index
    int8_t prevDepth = aux.depth - 1;
    uint8_t pathIndex = DeterminePathIndex(prevDepth);

    // Accumulate the contribution if
    if(isPathRayNEEOff   || // We hit a light with a path ray while NEE is off
       isPathRayAsMISRay || // We hit a light with a path ray while MIS option is enabled
       isCorrectNEERay   || // We hit the correct light as a NEE ray while NEE is on
       isCameraRay       || // We hit as a camera ray which should not be culled when NEE is on
       isSpecularPathRay)   // We hit as spec ray which did not launched any NEE rays thus it should contribute
    {
        // Accumulate the pixel
        ImageAccumulatePixel(renderState.gImage,
                             aux.pixelIndex,
                             Vector4f(total, 1.0f));

        // Also back propagate this radiance to the path nodes
        if(aux.type != RayType::CAMERA_RAY &&
           // If current path is the first vertex in the chain skip
           pathIndex != 0)
        {
            // Accumulate Radiance from 2nd vertex (including this vertex) away
            uint8_t prevPathIndex = pathIndex - 1;
            gLocalPathNodes[prevPathIndex].AccumRadianceDownChain(total, gLocalPathNodes);
        }
    }

    // Accumulate to the write tree as well for direct contribution
    // Do this only if path ray hits the light
    // regardless of nee is on or off
    if(aux.type == RayType::PATH_RAY ||
       aux.type == RayType::SPECULAR_PATH_RAY)
    {
        uint32_t dTreeIndex = gLocalPathNodes[pathIndex].nearestDTreeIndex;
        DTreeGPU& dWriteTree = renderState.gWriteDTrees[dTreeIndex];

        dWriteTree.AddRadianceToLeaf(r.getDirection(),
                                     Utility::RGBToLuminance(emission),
                                     true);
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

    // Path Memory
    const uint32_t pathStartIndex = aux.pathIndex * renderState.maximumPathNodePerRay;
    PathGuidingNode* gLocalPathNodes = renderState.gPathNodes + pathStartIndex;

    // TODO: change this currently only first strategy is sampled
    static constexpr int PATH_RAY_INDEX = 0;
    static constexpr int NEE_RAY_INDEX  = 1;
    // Helper Class for better code readability
    OutputWriter<RayAuxPPG> outputWriter(gOutBoundKeys,
                                         gOutRays,
                                         gOutRayAux,
                                         maxOutRay);

    // Inputs
    // Current Ray
    const RayF& r = ray.ray;
    // Hit Position
    Vector3 position = r.AdvancedPos(ray.tMax);
    // Wi (direction is swapped as if it is coming out of the surface)
    Vector3 wi = -(r.getDirection().Normalize());
    // Current ray's medium
    const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);

    // Nearest DTree
    // Find nearest DTree
    uint32_t dTreeIndex = UINT32_MAX;
    //dTreeIndex = 0;
    renderState.gStree->AcquireNearestDTree(dTreeIndex, position);
    assert(dTreeIndex != UINT32_MAX);

    // Check Material Sample Strategy
    uint32_t sampleCount = maxOutRay;
    // Check Material's specularity;
    float specularity = MGroup::Specularity(surface, gMatData, matIndex);
    bool isSpecularMat = (specularity >= TracerConstants::SPECULAR_TRESHOLD);

    // If NEE ray hits to this material
    // just skip since this is not a light material
    if(aux.type == RayType::NEE_RAY) return;

    // Calculate Transmittance factor of the medium
    // And reduce the radiance wrt the medium transmittance
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
        ImageAccumulatePixel(renderState.gImage,
                             aux.pixelIndex,
                             Vector4f(total, 1.0f));

        // Accumulate this to the paths as well
        if(total.HasNaN()) printf("NAN Found emissive!!!\n");
        gLocalPathNodes[aux.depth].AccumRadianceDownChain(total, gLocalPathNodes);
    }

    // If this material does not require to have any samples just quit
    if(sampleCount == 0) return;

    bool shouldLaunchMISRay = false;
    // ===================================== //
    //              NEE PORTION              //
    // ===================================== //
    // Don't launch NEE if not requested
    // or material is highly specular
    if(renderState.nee && !isSpecularMat)
    {
        float pdfLight, lDistance;
        HitKey matLight;
        Vector3 lDirection;
        uint32_t lightIndex;
        Vector3f neeReflectance = Zero3;
        if(renderState.gLightSampler->SampleLight(matLight,
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
        shouldLaunchMISRay = (renderState.directLightMIS &&
                              // Check if light can be sampled (meaning it is not a
                              // dirac delta light (point light spot light etc.)
                              renderState.gLightList[lightIndex]->CanBeSampled());

        float pdfNEE = pdfLight;
        // Weight the NEE if using MIS
        if(shouldLaunchMISRay)
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
            // Generate Ray
            RayF rayNEE = RayF(lDirection, position);
            rayNEE.AdvanceSelf(MathConstants::Epsilon);
            RayReg rayOut;
            rayOut.ray = rayNEE;
            rayOut.tMin = 0.0f;
            rayOut.tMax = lDistance;
            // Aux
            RayAuxPPG auxOut = aux;
            auxOut.radianceFactor = neeRadianceFactor;
            auxOut.endpointIndex = lightIndex;
            auxOut.type = RayType::NEE_RAY;
            auxOut.prevPDF = NAN;
            auxOut.depth++;
            outputWriter.Write(NEE_RAY_INDEX, rayOut, auxOut, matLight);
        }
    }

    // ==================================== //
    //             BxDF PORTION             //
    // ==================================== //
    // Sample a path from material
    RayF rayPath; float pdfPath; const GPUMediumI* outM = &m;
    Vector3f reflectance;
    // Sample a path using SDTree
    if(!isSpecularMat)
    {
        constexpr float BxDF_DTreeSampleRatio = 0.5f;
        // Sample a chance
        float xi = GPUDistribution::Uniform<float>(rng);
        const DTreeGPU& dReadTree = renderState.gReadDTrees[dTreeIndex];

        float pdfBxDF, pdfTree;
        if(xi < BxDF_DTreeSampleRatio)
        {
            // Sample using BxDF
            reflectance = MGroup::Sample(// Outputs
                                         rayPath, pdfBxDF, outM,
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
            pdfTree = dReadTree.Pdf(rayPath.getDirection());
        }
        else
        {
            // Sample a path using SDTree
            Vector3f direction = dReadTree.Sample(pdfTree, rng);
            direction.NormalizeSelf();
            // Calculate BxDF
            reflectance = MGroup::Evaluate(// Input
                                           direction,
                                           wi,
                                           position,
                                           m,
                                           //
                                           surface,
                                           // Constants
                                           gMatData,
                                           matIndex);

            pdfBxDF = MGroup::Pdf(direction,
                                  wi,
                                  position,
                                  m,
                                  //
                                  surface,
                                  gMatData,
                                  matIndex);

            // Generate a ray using the values
            rayPath = RayF(direction, position);
            rayPath.AdvanceSelf(MathConstants::Epsilon);
        }
        // Pdf Average
        pdfPath = BxDF_DTreeSampleRatio          * pdfBxDF +
                  (1.0f - BxDF_DTreeSampleRatio) * pdfTree;




        // DEBUG
        if(isnan(pdfPath) || isnan(pdfBxDF) || isnan(pdfTree))
            printf("[%s] NAN PDF = % f = w * %f + (1.0f - w) * %f, w: % f\n",
                   (xi < BxDF_DTreeSampleRatio) ? "BxDF": "Tree",
                   pdfPath, pdfBxDF, pdfTree, BxDF_DTreeSampleRatio);
        if(pdfPath != 0.0f && rayPath.getDirection().HasNaN())
            printf("[%s] NAN DIR %f, %f, %f\n",
                    (xi < BxDF_DTreeSampleRatio) ? "BxDF" : "Tree",
                    rayPath.getDirection()[0],
                    rayPath.getDirection()[1],
                    rayPath.getDirection()[2]);
        if(reflectance.HasNaN())
            printf("[%s] NAN REFL %f %f %f\n",
                   (xi < BxDF_DTreeSampleRatio) ? "BxDF" : "Tree",
                   reflectance[0],
                   reflectance[1],
                   reflectance[2]);

        //if(isnan(pdfPath) || isnan(pdfBxDF) || isnan(pdfTree) ||
        //   rayPath.getDirection().HasNaN() || reflectance.HasNaN())
        //    return;
    }
    else
    {
        // Specular Mat, Only sample using BxDF
        reflectance = MGroup::Sample(// Outputs
                                     rayPath, pdfPath, outM,
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
    pathRadianceFactor = (pdfPath == 0.0f) ? Zero3 : (pathRadianceFactor / pdfPath);

    if(pathRadianceFactor.HasNaN())
        printf("NAN PATH R: %f %f %f = {%f %f %f} * {%f %f %f} / %f  \n",
               pathRadianceFactor[0], pathRadianceFactor[1], pathRadianceFactor[2],
               radianceFactor[0], radianceFactor[1], radianceFactor[2],
               reflectance[0], reflectance[1], reflectance[2], pdfPath);

    // Check Russian Roulette
    float avgThroughput = pathRadianceFactor.Dot(Vector3f(0.333f));
    bool terminateRay = ((aux.depth > renderState.rrStart) &&
                         TracerFunctions::RussianRoulette(pathRadianceFactor, avgThroughput, rng));

    // Do not terminate rays ever for specular mats
    if((!terminateRay || isSpecularMat) &&
        // Do not waste rays on zero radiance paths
        pathRadianceFactor != ZERO_3)
    {
        // Ray
        RayReg rayOut;
        rayOut.ray = rayPath;
        rayOut.tMin = 0.0f;
        rayOut.tMax = FLT_MAX;
        // Aux
        RayAuxPPG auxOut = aux;
        auxOut.mediumIndex = static_cast<uint16_t>(outM->GlobalIndex());
        auxOut.radianceFactor = pathRadianceFactor;
        auxOut.type = (isSpecularMat) ? RayType::SPECULAR_PATH_RAY : RayType::PATH_RAY;
        // Save BxDF pdf if we hit a light
        // When we hit a light we will
        auxOut.prevPDF = pdfPath;
        auxOut.depth++;
        // Write
        outputWriter.Write(PATH_RAY_INDEX, rayOut, auxOut);
    }

    //// Add the sample towards that position on the tree
    //DTreeGPU& dWriteTree = renderState.gWriteDTrees[dTreeIndex];
    //dWriteTree.AddSampleToLeaf(rayPath.getDirection());

    // Record this intersection on path chain
    uint8_t prevPathIndex = DeterminePathIndex(aux.depth - 1);
    uint8_t curPathIndex = DeterminePathIndex(aux.depth);

    PathGuidingNode node;
    //printf("WritingNode PC:(%u %u) W:(%f, %f, %f) RF:(%f, %f, %f) Path: %u DT %u\n",
    //       static_cast<uint32_t>(prevDepth), static_cast<uint32_t>(currentDepth),
    //       position[0], position[1], position[2],
    //       pathRadianceFactor[0], pathRadianceFactor[1], pathRadianceFactor[2],
    //       aux.pathIndex, dTreeIndex);
    node.prevNext[1] = PathGuidingNode::InvalidIndex;
    node.prevNext[0] = prevPathIndex;
    node.worldPosition = position;
    node.nearestDTreeIndex = dTreeIndex;
    node.radFactor = pathRadianceFactor;
    node.totalRadiance = Zero3;
    gLocalPathNodes[curPathIndex] = node;
    // Set Previous Path node's next index
    if(prevPathIndex != PathGuidingNode::InvalidIndex)
        gLocalPathNodes[prevPathIndex].prevNext[1] = curPathIndex;

    // All Done!
}