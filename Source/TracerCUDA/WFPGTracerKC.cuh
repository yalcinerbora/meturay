#pragma once

#include "RayAuxStruct.cuh"

#include "GPULightI.h"
#include "GPUMediumI.h"
#include "GPUDirectLightSamplerI.h"
#include "PathNode.cuh"
#include "RayStructs.h"
#include "ImageStructs.h"
#include "WorkOutputWriter.cuh"
#include "WFPGCommon.h"
#include "GPUCameraI.h"
#include "GPUMetaSurfaceGenerator.h"
#include "ProductSampler.cuh"
#include "SVOOptiXRadianceBuffer.cuh"

#include "TracerFunctions.cuh"
#include "TracerConstants.h"

#include "RayLib/RandomColor.h"

#include "AnisoSVO.cuh"

#include "GPUBlockPWCDistribution.cuh"
#include "GPUBlockPWLDistribution.cuh"
#include "BlockTextureFilter.cuh"

#include "RecursiveConeTrace.cuh"
#include "RNGConstant.cuh"

#include "GPUCameraPinhole.cuh"


static constexpr uint32_t INVALID_BIN_ID = std::numeric_limits<uint32_t>::max();

struct WFPGTracerGlobalState
{
    // Output Samples
    CamSampleGMem<Vector4f>         gSamples;
    // Light Related
    const GPULightI**               gLightList;
    uint32_t                        totalLightCount;
    const GPUDirectLightSamplerI*   gLightSampler;
    // Medium Related
    const GPUMediumI**              mediumList;
    uint32_t                        totalMediumCount;
    // Path Guiding Related
    AnisoSVOctreeGPU                svo;
    // Path Related
    WFPGPathNode*                   gPathNodes;
    uint32_t                        maximumPathNodePerRay;
    // Options
    // Path Guiding
    bool                            skipPG;
    // Options for NEE
    bool                            directLightMIS;
    bool                            nee;
    // Russian Roulette
    int                             rrStart;
};

struct WFPGTracerLocalState
{
    bool    emptyPrimitive;
};

__device__ inline
uint8_t DeterminePathIndexWFPG(uint8_t depth)
{
    return depth;
}

template <class EGroup>
__device__ inline
void WFPGTracerBoundaryWork(// Output
                            HitKey* gOutBoundKeys,
                            RayGMem* gOutRays,
                            RayAuxWFPG* gOutRayAux,
                            const uint32_t maxOutRay,
                            // Input as registers
                            const RayReg& ray,
                            const RayAuxWFPG& aux,
                            const typename EGroup::Surface& surface,
                            const RayId rayId,
                            // I-O
                            WFPGTracerLocalState& localState,
                            WFPGTracerGlobalState& renderState,
                            RNGeneratorGPUI& rng,
                            // Constants
                            const typename EGroup::GPUType& gLight)
{
    using GPUType = typename EGroup::GPUType;

    // Current Path
    const uint32_t pathStartIndex = aux.sampleIndex * renderState.maximumPathNodePerRay;
    WFPGPathNode* gLocalPathNodes = renderState.gPathNodes + pathStartIndex;

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
        Vector3 position = surface.WorldPosition();
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
    Vector3 position = surface.WorldPosition();
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
    uint8_t prevPathIndex = DeterminePathIndexWFPG(prevDepth);

    // Accumulate the contribution if
    if(isPathRayNEEOff   || // We hit a light with a path ray while NEE is off
       isPathRayAsMISRay || // We hit a light with a path ray while MIS option is enabled
       isCorrectNEERay   || // We hit the correct light as a NEE ray while NEE is on
       isCameraRay       || // We hit as a camera ray which should not be culled when NEE is on
       isSpecularPathRay)   // We hit as spec ray which did not launched any NEE rays thus it should contribute
    {
        // Accumulate the pixel
        AccumulateRaySample(renderState.gSamples,
                            aux.sampleIndex,
                            Vector4f(total, 1.0f));

        // Also back propagate this radiance to the path nodes
        if(aux.type != RayType::CAMERA_RAY)
           // &&
           //// If current path is the first vertex in the chain skip
           //pathIndex != 0)
        {
            // Accumulate Radiance from 2nd vertex (including this vertex) away
            // S-> surface
            // C-> camera
            // L-> light
            //
            // C --- S --- S --- S --- L
            //                   |     ^
            //                   |     We are here (pathIndex points here)
            //                   v
            //                  we should accum-down from here
            gLocalPathNodes[prevPathIndex].AccumRadianceDownChain(total, gLocalPathNodes);
        }
    }
}

template <class MGroup>
__device__ inline
void WFPGTracerPathWork(// Output
                        HitKey* gOutBoundKeys,
                        RayGMem* gOutRays,
                        RayAuxWFPG* gOutRayAux,
                        const uint32_t maxOutRay,
                        // Input as registers
                        const RayReg& ray,
                        const RayAuxWFPG& aux,
                        const typename MGroup::Surface& surface,
                        const RayId rayId,
                        // I-O
                        WFPGTracerLocalState& localState,
                        WFPGTracerGlobalState& renderState,
                        RNGeneratorGPUI& rng,
                        // Constants
                        const typename MGroup::Data& gMatData,
                        const HitKey::Type matIndex)
{
    static constexpr Vector3 ZERO_3 = Zero3;

    // Path Memory
    const uint32_t pathStartIndex = aux.sampleIndex * renderState.maximumPathNodePerRay;
    WFPGPathNode* gLocalPathNodes = renderState.gPathNodes + pathStartIndex;

    // TODO: change this currently only first strategy is sampled
    static constexpr int PATH_RAY_INDEX = 0;
    static constexpr int NEE_RAY_INDEX  = 1;
    // Helper Class for better code readability
    OutputWriter<RayAuxWFPG> outputWriter(gOutBoundKeys,
                                          gOutRays,
                                          gOutRayAux,
                                          maxOutRay);

    // Inputs
    // Current Ray
    const RayF& r = ray.ray;
    // Hit Position
    Vector3 position = surface.WorldPosition();
    // Wi (direction is swapped as if it is coming out of the surface)
    Vector3 wi = -(r.getDirection().Normalize());
    // Current ray's medium
    const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);

    // Check Material Sample Strategy
    uint32_t sampleCount = maxOutRay;
    // Check Material's specularity;
    float specularity = MGroup::Specularity(surface, gMatData, matIndex);
    bool isSpecularMat = (specularity >= TracerConstants::SPECULAR_THRESHOLD);

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
        AccumulateRaySample(renderState.gSamples,
                            aux.sampleIndex,
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

            // Check if mis ray should be sampled
            shouldLaunchMISRay = (renderState.directLightMIS &&
                                  // Check if light can be sampled (meaning it is not a
                                  // dirac delta light (point light spot light etc.)
                                  renderState.gLightList[lightIndex]->CanBeSampled());
        }
        else shouldLaunchMISRay = false;

        // Weight the NEE if using MIS
        float pdfNEE = pdfLight;
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
            rayNEE.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);
            RayReg rayOut;
            rayOut.ray = rayNEE;
            rayOut.tMin = 0.0f;
            rayOut.tMax = lDistance;
            // Aux
            RayAuxWFPG auxOut = aux;
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
        // Other kernel already combined with MIS
        // just evaluate
        if(renderState.skipPG)
        {
            // Sample using BxDF
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
        else
        {
            Vector2f uv = Vector2f(aux.guideDir[0], aux.guideDir[1]);
            Vector3f dirZUp = Utility::CocentricOctohedralToDirection(uv);
            Vector3f direction = Vector3f(dirZUp[1], dirZUp[2], dirZUp[0]);
            pdfPath = aux.guidePDF;

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

            rayPath = RayF(direction, position);
            rayPath.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);
        }


        //float BxDF_GuideSampleRatio = (renderState.skipPG) ? 0.0f : 0.5f;
        //BxDF_GuideSampleRatio = (renderState.purePG) ? 1.0f : BxDF_GuideSampleRatio;
        //float xi = rng.Uniform();

        //float misWeight;
        //bool selectedPDFZero = false;
        //float pdfBxDF, pdfGuide;
        //bool BxDFSelected = (xi >= BxDF_GuideSampleRatio);
        //if(BxDFSelected)
        //{
        //    // Sample using BxDF
        //    reflectance = MGroup::Sample(// Outputs
        //                                 rayPath, pdfBxDF, outM,
        //                                 // Inputs
        //                                 wi,
        //                                 position,
        //                                 m,
        //                                 //
        //                                 surface,
        //                                 // I-O
        //                                 rng,
        //                                 // Constants
        //                                 gMatData,
        //                                 matIndex,
        //                                 0);
        //    pdfGuide = aux.guidePDF;
        //    misWeight = TracerFunctions::BalanceHeuristic(1 - BxDF_GuideSampleRatio, pdfBxDF,
        //                                                  BxDF_GuideSampleRatio, pdfGuide);
        //    BxDF_GuideSampleRatio = 1.0f - BxDF_GuideSampleRatio;
        //    selectedPDFZero = (pdfBxDF == 0.0f);
        //    printf("noooo");
        //}
        //else
        //{
        //    // Sample a path from the pre-sampled UV
        //    // uv coordinates to spherical coordinates
        //    Vector2f uv = Vector2f(aux.guideDir[0], aux.guideDir[1]);
        //    Vector3f dirZUp = Utility::CocentricOctohedralToDirection(uv);
        //    Vector3f direction = Vector3f(dirZUp[1], dirZUp[2], dirZUp[0]);
        //    pdfGuide = aux.guidePDF;

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

        //    pdfBxDF = MGroup::Pdf(direction,
        //                          wi,
        //                          position,
        //                          m,
        //                          //
        //                          surface,
        //                          gMatData,
        //                          matIndex);

        //    // Generate a ray using the values
        //    rayPath = RayF(direction, position);
        //    rayPath.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);

        //    misWeight = TracerFunctions::BalanceHeuristic(BxDF_GuideSampleRatio, pdfGuide,
        //                                                  1 - BxDF_GuideSampleRatio, pdfBxDF);
        //    selectedPDFZero = (pdfGuide == 0.0f);
        //}
        //// One-sample MIS Using Balance Heuristic
        //if(!renderState.skipPG &&
        //   !renderState.purePG)
        //{
        //    pdfPath = (BxDFSelected) ? pdfBxDF : pdfGuide;
        //    pdfPath = BxDF_GuideSampleRatio * pdfPath / misWeight;
        //    pdfPath = selectedPDFZero ? 0.0f : pdfPath;
        //}
        //else if(renderState.purePG)
        //    pdfPath = pdfGuide;
        //else
        //    pdfPath = pdfBxDF;

        // DEBUG
        //if(isnan(pdfPath) || isnan(pdfBxDF) || isnan(pdfGuide))
        //if(isnan(pdfPath) || isnan(pdfBxDF) || isnan(pdfGuide))
        //    printf("[%s] NAN PDF = % f = w * %f + (1.0f - w) * %f, w: % f\n",
        //           (BxDFSelected) ? "BxDF": "SVO",
        //           pdfPath, pdfBxDF, pdfGuide, BxDF_GuideSampleRatio);
        //if(pdfPath != 0.0f && rayPath.getDirection().HasNaN())
        //    printf("[%s] NAN DIR %f, %f, %f\n",
        //            (BxDFSelected) ? "BxDF" : "SVO",
        //            rayPath.getDirection()[0],
        //            rayPath.getDirection()[1],
        //            rayPath.getDirection()[2]);
        //if(reflectance.HasNaN())
        //    printf("[%s] NAN REFL %f %f %f\n",
        //           (BxDFSelected) ? "BxDF" : "SVO",
        //           reflectance[0],
        //           reflectance[1],
        //           reflectance[2]);

        //if(isnan(pdfPath) || isnan(pdfBxDF) || isnan(pdfGuide) ||
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
        RayAuxWFPG auxOut = aux;
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
    // Record this intersection on path chain
    uint8_t prevPathIndex = DeterminePathIndexWFPG(aux.depth - 1);
    uint8_t curPathIndex = DeterminePathIndexWFPG(aux.depth);

    WFPGPathNode node;
    //printf("WritingNode PC:(%u %u) W:(%f, %f, %f) RF:(%f, %f, %f) Path: %u\n",
    //       static_cast<uint32_t>(prevPathIndex), static_cast<uint32_t>(curPathIndex),
    //       position[0], position[1], position[2],
    //       pathRadianceFactor[0], pathRadianceFactor[1], pathRadianceFactor[2],
    //       aux.pathIndex);
    node.prevNext[1] = WFPGPathNode::InvalidIndex;
    node.prevNext[0] = prevPathIndex;
    node.worldPosition = position;
    node.SetNormal(surface.WorldGeoNormal());
    // Unlike other techniques that holds incoming radiance
    // WFPG holds outgoing radiance. To calculate that, wee need to store
    // previous paths throughput
    node.radFactor = aux.radianceFactor;
    node.totalRadiance = Zero3;
    gLocalPathNodes[curPathIndex] = node;
    // Set Previous Path node's next index
    if(prevPathIndex != WFPGPathNode::InvalidIndex)
        gLocalPathNodes[prevPathIndex].prevNext[1] = curPathIndex;

    // All Done!
}

template <class EGroup>
__device__ inline
void WFPGTracerDebugBWork(// Output
                          HitKey* gOutBoundKeys,
                          RayGMem* gOutRays,
                          RayAuxWFPG* gOutRayAux,
                          const uint32_t maxOutRay,
                          // Input as registers
                          const RayReg& ray,
                          const RayAuxWFPG& aux,
                          const typename EGroup::Surface& surface,
                          const RayId rayId,
                          // I-O
                          WFPGTracerLocalState& localState,
                          WFPGTracerGlobalState& renderState,
                          RNGeneratorGPUI& rng,
                          // Constants
                          const typename EGroup::GPUType& gLight)
{
    const AnisoSVOctreeGPU& svo = renderState.svo;
    // Helper Class for better code readability
    OutputWriter<RayAuxWFPG> outputWriter(gOutBoundKeys,
                                          gOutRays,
                                          gOutRayAux,
                                          maxOutRay);

    // Only Direct Hits from camera are used
    // In debugging
    if(aux.depth != 1) return;
    // Hit Position
    Vector3 position = surface.WorldPosition();

    // Query SVO find a leaf
    uint32_t svoLeafIndex;
    bool found = svo.NearestNodeIndex(svoLeafIndex, position,
                                      svo.LeafDepth(), true);
    Vector3f locColor = (found) ? Utility::RandomColorRGB(svoLeafIndex)
                                : Vector3f(0.0f);
    // Accumulate the pixel
    AccumulateRaySample(renderState.gSamples,
                        aux.sampleIndex,
                        Vector4f(locColor, 1.0f));
}

template <class MGroup>
__device__ inline
void WFPGTracerDebugWork(// Output
                         HitKey* gOutBoundKeys,
                         RayGMem* gOutRays,
                         RayAuxWFPG* gOutRayAux,
                         const uint32_t maxOutRay,
                         // Input as registers
                         const RayReg& ray,
                         const RayAuxWFPG& aux,
                         const typename MGroup::Surface& surface,
                         const RayId rayId,
                         // I-O
                         WFPGTracerLocalState& localState,
                         WFPGTracerGlobalState& renderState,
                         RNGeneratorGPUI& rng,
                         // Constants
                         const typename MGroup::Data& gMatData,
                         const HitKey::Type matIndex)
{
    const AnisoSVOctreeGPU& svo = renderState.svo;
    // Helper Class for better code readability
    OutputWriter<RayAuxWFPG> outputWriter(gOutBoundKeys,
                                          gOutRays,
                                          gOutRayAux,
                                          maxOutRay);
    // Only Direct Hits from camera are used
    // In debugging
    if(aux.depth != 1) return;
    // Hit Position
    Vector3 position = surface.WorldPosition();

    // Query SVO find a leaf
    uint32_t svoLeafIndex;
    bool found = svo.NearestNodeIndex(svoLeafIndex, position,
                                      svo.LeafDepth(), true);
    Vector3f locColor = (found) ? Utility::RandomColorRGB(svoLeafIndex)
                                : Vector3f(0.0f);
    // Accumulate the pixel
    AccumulateRaySample(renderState.gSamples,
                        aux.sampleIndex,
                        Vector4f(locColor, 1.0f));
}

__device__ inline
float ReadInterpolatedRadiance(const Vector3f& hitPos,
                               const Vector3f& direction, float coneApterture,
                               const AnisoSVOctreeGPU& svo)
{
    float queryVoxelSize = svo.LeafVoxelSize();
    Vector3f offsetPos = hitPos + direction.Normalize() * queryVoxelSize * 0.5f;


    // Interp values for currentLevel
    Vector3f lIndex = (offsetPos - svo.OctreeAABB().Min()) / queryVoxelSize;
    Vector3f lIndexInt;
    Vector3f lIndexFrac = Vector3f(modff(lIndex[0], &(lIndexInt[0])),
                                   modff(lIndex[1], &(lIndexInt[1])),
                                   modff(lIndex[2], &(lIndexInt[2])));
    Vector3ui denseIndex = Vector3ui(lIndexInt);

    Vector3ui inc = Vector3ui((lIndexFrac[0] < 0.5f) ? -1 : 0,
                              (lIndexFrac[1] < 0.5f) ? -1 : 0,
                              (lIndexFrac[2] < 0.5f) ? -1 : 0);

    // Calculate Interp start index and values
    denseIndex += inc;
    Vector3f interpValues = (lIndexFrac + Vector3f(0.5f));
    interpValues -= interpValues.Floor();

    // Get 8x value
    float irradiance[8];
    for(int i = 0; i < 8; i++)
    {
        Vector3ui curIndex = Vector3ui(((i >> 0) & 0b1) ? 1 : 0,
                                       ((i >> 1) & 0b1) ? 1 : 0,
                                       ((i >> 2) & 0b1) ? 1 : 0);
        Vector3ui voxIndex = denseIndex + curIndex;
        uint64_t voxelMorton = MortonCode::Compose3D<uint64_t>(voxIndex);

        uint32_t nodeId;
        bool found = svo.Descend(nodeId, voxelMorton, svo.LeafDepth());

        irradiance[i] = (found) ? static_cast<float>(svo.ReadRadiance(direction, coneApterture,
                                                                      nodeId, true))
                                : 0.0f;
    }

    float x0 = HybridFuncs::Lerp(irradiance[0], irradiance[1], interpValues[0]);
    float x1 = HybridFuncs::Lerp(irradiance[2], irradiance[3], interpValues[0]);
    float x2 = HybridFuncs::Lerp(irradiance[4], irradiance[5], interpValues[0]);
    float x3 = HybridFuncs::Lerp(irradiance[6], irradiance[7], interpValues[0]);
    float y0 = HybridFuncs::Lerp(x0, x1, interpValues[1]);
    float y1 = HybridFuncs::Lerp(x2, x3, interpValues[1]);
    return HybridFuncs::Lerp(y0, y1, interpValues[2]);
}

template <int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
struct KCTraceSVOSharedMem
{
    static constexpr size_t MAX_CAM_CLASS_SIZE = 256;
    using ConeTracer = BatchConeTracer<THREAD_PER_BLOCK, X, Y>;
    using ConeTraceMem = typename ConeTracer::TempStorage;

    ConeTraceMem        sConeTraceMem;
    Byte                sSubCamMemory[MAX_CAM_CLASS_SIZE];
    const GPUCameraI*   sSubCam;
    Vector3f            sPosition;
    Vector2f            sTMinMax;
    float               sConeAperture;
};

__device__
inline float GenSolidAngle(const GPUCameraI& gCamera,
                           const Vector2i& resolution)
{
    // https://math.stackexchange.com/questions/1281112/how-to-calculate-a-solid-angle-in-steradians-given-only-horizontal-beam-angle
    Vector2f f = gCamera.FoV();
    Vector2f res = Vector2f(resolution);
    float solidAngle = 4.0f * asin(sin(f[0] * 0.5) * sin(f[1] * 0.5));
    // Case for x,y > Pi
    if(f[0] > MathConstants::Pi || f[1] > MathConstants::Pi)
        solidAngle = 4.0f * MathConstants::Pi - solidAngle;
    // Assuming pixels are square here
    float totalPixCount = res.Multiply();
    return solidAngle / totalPixCount;
}

static constexpr int FALSE_COLOR_MASK_COUNT = 3;
__device__ static const Vector3f SVO_LEVEL_FALSE_COLOR_MASK[FALSE_COLOR_MASK_COUNT] =
{
    Vector3f(1.0f, 0.0f, 0.0f),
    Vector3f(0.0f, 1.0f, 0.0f),
    Vector3f(0.0f, 0.0f, 1.0f)
};

__device__ inline
Vector4f CalcColorSVO(WFPGRenderMode mode,
                      const AnisoSVOctreeGPU& svo,
                      Vector3f rayDir, float coneSolidAngle,
                      uint32_t nodeId, uint32_t nodeLevel)
{
    bool isLeaf = (nodeLevel == svo.LeafDepth());
    Vector3f result = Zero3f;
    if(mode == WFPGRenderMode::SVO_FALSE_COLOR)
    {
        // Saturate using level
        Vector3f color = Utility::RandomColorRGB(nodeId);
        uint32_t levelDiff = svo.LeafDepth() - nodeLevel;
        Vector3f levelColor = Utility::RandomColorRGB(levelDiff);
        //Vector3f levelColor = SVO_LEVEL_FALSE_COLOR_MASK[levelDiff % FALSE_COLOR_MASK_COUNT];

        color = (color + levelColor) * 0.5f;
        color = (nodeId != UINT32_MAX) ? color : Zero3f;
        result = color;
    }
    // Payload Display Mode
    else if(mode == WFPGRenderMode::SVO_RADIANCE)
    {
        //Vector3f hitPos = ray.ray.getPosition() + rayDir.Normalize() * currentTMax;
        //float radianceF = ReadInterpolatedRadiance(hitPos, rayDir,
        //                                           params.pixelAperture,
        //                                           svo);

        float radiance = svo.ReadRadiance(rayDir, coneSolidAngle,
                                          nodeId, isLeaf);
        result = Vector3f(radiance);
    }
    else if(nodeId == UINT32_MAX)
        return Vector4f(result, 1.0f);
    else if(mode == WFPGRenderMode::SVO_NORMAL)
    {
        float stdDev;
        Vector3f normal = svo.DebugReadNormal(stdDev, nodeId, isLeaf);

        // Voxels are two sided show the normal for the current direction
        normal = (normal.Dot(rayDir) >= 0.0f) ? normal : -normal;

        // Convert normal to 0-1 range
        normal += Vector3f(1.0f);
        normal *= Vector3f(0.5f);
        return Vector4f(normal, stdDev);
    }
    return Vector4f(result, 1.0f);
}


template <int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
__device__
inline void TraceSVO(// Output
                     CamSampleGMem<Vector4f> gSamples,
                     // Input
                     const GPUCameraI* gCamera,
                     // Constants
                     const AnisoSVOctreeGPU& svo,
                     WFPGRenderMode mode,
                     uint32_t maxQueryLevelOffset,

                     const Vector2i& totalPixelCount,
                     const Vector2i& totalSegments)
{
    const uint32_t totalBlockCount = totalSegments.Multiply();

    // TODO change this to dynamic
    using SharedMemType = KCTraceSVOSharedMem<THREAD_PER_BLOCK, X, Y>;
    using ConeTracer = typename SharedMemType::ConeTracer;
    // SharedMemory
    // Change the type of the shared memory
    extern __shared__ Byte sharedMemRAW[];
    SharedMemType* shMem = reinterpret_cast<SharedMemType*>(sharedMemRAW);

    ConeTracer batchedConeTracer(shMem->sConeTraceMem, svo);
    RNGConstantGPU rng(0.0f);

    // Block stride loop
    for(int32_t blockId = blockIdx.x;
        blockId < totalBlockCount; blockId += gridDim.x)
    {
        int32_t localThreadId = threadIdx.x;
        // Calculate image segment size offset for this block
        Vector2i segmentSize = Vector2i(X, Y);
        Vector2i segmentId2D = Vector2i(blockId % totalSegments[0],
                                        blockId / totalSegments[0]);

        // Leader Generates Camera etc.
        if(localThreadId == 0)
        {
            shMem->sSubCam = gCamera->GenerateSubCamera(shMem->sSubCamMemory,
                                                        SharedMemType::MAX_CAM_CLASS_SIZE,
                                                        segmentId2D,
                                                        totalSegments);
            // This should only work pinhole style cameras
            // Main thread chooses the starting position
            // min max
            RayReg ray; Vector2f pixCoords;
            shMem->sSubCam->GenerateRay(ray, pixCoords, Vector2i(0), segmentSize,
                                        rng, false);
            shMem->sPosition = ray.ray.getPosition();
            shMem->sTMinMax = Vector2f(ray.tMin, ray.tMax);
            shMem->sConeAperture = GenSolidAngle(*gCamera, totalPixelCount);
        }
        __syncthreads();

        const GPUCameraI* sCam = shMem->sSubCam;
        // Gen proj functor
        auto ProjectionFunc = [sCam, &rng](const Vector2i& localPixelId,
                                           const Vector2i& segmentSize)
        {
            //
            RayReg ray; Vector2f uv;
            sCam->GenerateRay(ray, uv, localPixelId, segmentSize,
                              rng, false);
            return ray.ray.getDirection();
        };
        // Gen wrap functor
        //auto WrapFunc = [](const Vector2i& pixelId,
        //                   const Vector2i& segmentSize)
        //{
        //    return pixelId.Clamp(Vector2i(0), segmentSize - 1);
        //};

        float tMin[ConeTracer::DATA_PER_THREAD];
        bool  isLeaf[ConeTracer::DATA_PER_THREAD];
        uint32_t nodeIndex[ConeTracer::DATA_PER_THREAD];
        Vector3f rayDir[ConeTracer::DATA_PER_THREAD];

        //batchedConeTracer.RecursiveConeTraceRay(rayDir, tMin, isLeaf,
        //                                        nodeIndex,
        //                                        // Inputs
        //                                        shMem->sPosition,
        //                                        shMem->sTMinMax,
        //                                        shMem->sConeAperture,
        //                                        0, 1, maxQueryLevelOffset,
        //                                        ProjectionFunc,
        //                                        WrapFunc);
        batchedConeTracer.BatchedConeTraceRay(rayDir, tMin, isLeaf,
                                              nodeIndex,
                                              // Inputs
                                              shMem->sPosition,
                                              shMem->sTMinMax,
                                              shMem->sConeAperture,
                                              maxQueryLevelOffset,
                                              ProjectionFunc);

        // Write as color
        for(int i = 0; i < ConeTracer::DATA_PER_THREAD; i++)
        {
            uint32_t nodeLevel = svo.NodeLevel(nodeIndex[i], isLeaf[i]);
            Vector4f locColor = CalcColorSVO(mode, svo, rayDir[i],
                                             shMem->sConeAperture,
                                             nodeIndex[i], nodeLevel);
            //Vector4f locColor = Vector4f(0.0f, 0.0f, 10.0f, 1.0f);
            //// Octree Display Mode
            //if(mode == WFPGRenderMode::SVO_FALSE_COLOR)
            //    locColor = (nodeIndex[i] != UINT32_MAX) ? Vector4f(Utility::RandomColorRGB(nodeIndex[i]), 1.0f)
            //                                            : Vector4f(Vector3f(0.0f), 1.0f);
            //// Payload Display Mode
            //else if(nodeIndex[i] == UINT32_MAX)
            //    locColor = Vector4f(1.0f, 0.0f, 1.0f, 1.0f);
            //else if(mode == WFPGRenderMode::SVO_RADIANCE)
            //{
            //    Vector3f hitPos = shMem->sPosition + rayDir[i].Normalize() * tMin[i];
            //    //float radianceF = ReadInterpolatedRadiance(hitPos, rayDir[i],
            //    //                                           shMem->sConeAperture,
            //    //                                           svo);

            //    half radiance = svo.ReadRadiance(rayDir[i], shMem->sConeAperture,
            //                                     nodeIndex[i], isLeaf[i]);
            //    float radianceF = radiance;
            //    //if(radiance != static_cast<half>(MRAY_HALF_MAX))
            //        locColor = Vector4f(Vector3f(radianceF), 1.0f);
            //}
            //else if(mode == WFPGRenderMode::SVO_NORMAL)
            //{
            //    float stdDev;
            //    Vector3f normal = svo.DebugReadNormal(stdDev, nodeIndex[i], isLeaf[i]);

            //    // Voxels are two sided show the normal for the current direction
            //    normal = (normal.Dot(rayDir[i]) >= 0.0f) ? normal : -normal;

            //    // Convert normal to 0-1 range
            //    normal += Vector3f(1.0f);
            //    normal *= Vector3f(0.5f);
            //    locColor = Vector4f(normal, stdDev);
            //}

            // Determine output
            int32_t linearInnerSampleId = THREAD_PER_BLOCK * i + localThreadId;
            if(linearInnerSampleId >= segmentSize.Multiply()) continue;

            // Actual write
            Vector2i innerSampleId = Vector2i(linearInnerSampleId % segmentSize[0],
                                              linearInnerSampleId / segmentSize[0]);
            Vector2i globalSampleId = segmentId2D * segmentSize + innerSampleId;
            int32_t globalLinearSampleId = (globalSampleId[1] * totalPixelCount[0] +
                                            globalSampleId[0]);
            // Create the img coords
            Vector2f imgCoords = Vector2f(globalSampleId) + Vector2f(0.5f);

            if(globalLinearSampleId < totalPixelCount.Multiply())
            {
                gSamples.gImgCoords[globalLinearSampleId] = imgCoords;
                gSamples.gValues[globalLinearSampleId] = locColor;
            }
            //else printf("fail\n");
        }
        __syncthreads();
    }
}

__device__ inline
uint32_t GenerateLeafIndex(uint32_t leafIndex)
{
    static constexpr uint32_t LAST_BIT_UINT32 = 31;
    uint32_t result = leafIndex;
    result |= (1u << LAST_BIT_UINT32);
    return result;
}

__device__ inline
uint32_t GenerateNodeIndex(uint32_t nodeIndex)
{
    return nodeIndex;
}

__device__ inline
bool ReadSVONodeId(uint32_t& nodeId, uint32_t packedData)
{
    static constexpr uint32_t LAST_BIT_UINT32 = 31;
    static constexpr uint32_t INDEX_MASK = (1u << LAST_BIT_UINT32) - 1;

    bool isLeaf = (packedData >> LAST_BIT_UINT32) & 0b1;
    nodeId = packedData & INDEX_MASK;
    return isLeaf;
}

template <class RNG>
__device__ inline
Vector3f DirIdToWorldDir(const Vector2ui& dirXY,
                         const Vector2ui& dimensions,
                         RNG& rng)
{
    assert(dirXY < dimensions);
    using namespace MathConstants;

    Vector2f xi = rng.Uniform2D();
    ////Vector2f xi = Vector2f(0.5f);

    // Generate st coords [0, 1] from integer coords
    Vector2f st = Vector2f(dirXY) + xi;
    st /= Vector2f(dimensions);
    Vector3f result = Utility::CocentricOctohedralToDirection(st);
    Vector3 dirYUp = Vector3(result[1], result[2], result[0]);
    return dirYUp;
}

__global__
static void KCInitializeSVOBins(// Outputs
                                uint32_t* gRayBindIds,
                                RayAuxWFPG* gRayAux,
                                // Inputs
                                const RayGMem* gRays,
                                const HitKey* gRayWorkKeys,
                                // Constants
                                HitKey boundaryMatKey,
                                AnisoSVOctreeGPU svo,
                                uint32_t rayCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < rayCount;
        threadId += (blockDim.x * gridDim.x))
    {
        const RayType rayType = gRayAux[threadId].type;
        RayReg ray = RayReg(gRays, threadId);
        HitKey key = gRayWorkKeys[threadId];

        bool unnecessaryRay = (rayType == RayType::NEE_RAY ||
                               key == HitKey::InvalidKey ||
                               key == boundaryMatKey);

        // Skip NEE rays
        // These either hit the light or not
        // For both cases it is better to skip them
        // Or if the ray is invalid
        // It happens when a material does not write a ray
        // to its available spot)
        if(unnecessaryRay)
        {
            gRayAux[threadId].binId = INVALID_BIN_ID;
            gRayBindIds[threadId] = INVALID_BIN_ID;
            continue;
        }

        // Hit Position
        Vector3 position = ray.ray.AdvancedPos(ray.tMax);

        // Find the leaf
        // Traverse all 8 neighbors of the ray hit position
        // due to numerical inaccuracies
        uint32_t leafIndex;
        bool found = svo.NearestNodeIndex(leafIndex, position,
                                          svo.LeafDepth(), true);

        // DEBUG
        if(!found) printf("Leaf Not found!\n");

        // Store the found leaf & increment the ray count
        // for that leaf
        if(found)
        {
            uint32_t binId = GenerateLeafIndex(leafIndex);
            // Write to ray payload
            gRayAux[threadId].binId = binId;
            // Write to other buffer for partitioning as well
            gRayBindIds[threadId] = binId;
        }
    }
}

__global__
static void KCWriteLeafRayAmounts(// Outputs
                                  AnisoSVOctreeGPU svo,
                                  // Inputs
                                  const uint32_t* gBinIds,
                                  const uint32_t* gBinCounts,
                                  // Constants
                                  uint32_t binCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < binCount;
        threadId += (blockDim.x * gridDim.x))
    {
        uint32_t binId = gBinIds[threadId];
        // Skip the invalid bin
        if(binId == INVALID_BIN_ID) continue;

        uint32_t rayCount = min(gBinCounts[threadId], UINT16_MAX);
        uint16_t rayCount16Bit = static_cast<uint16_t>(rayCount);

        uint32_t svoLeafId;
        bool isLeaf = ReadSVONodeId(svoLeafId, binId);
        assert(isLeaf);
        svo.SetLeafRayCount(svoLeafId, rayCount16Bit);
    }
}

__global__
static void KCCheckReducedSVOBins(// I-O
                                  RayAuxWFPG* gRayAux,
                                  // Constants
                                  AnisoSVOctreeGPU svo,
                                  uint32_t rayCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < rayCount;
        threadId += (blockDim.x * gridDim.x))
    {
        // Skip if this ray is not used
        if(gRayAux[threadId].binId == INVALID_BIN_ID)
            continue;

        uint32_t leafNodeId;
        ReadSVONodeId(leafNodeId, gRayAux[threadId].binId);

        bool isLeaf;
        uint32_t newBinId = svo.FindMarkedBin(isLeaf, leafNodeId);
        if(!isLeaf) gRayAux[threadId].binId = newBinId;
    }
}


// Shared Memory Class of the kernel below
template <int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
struct KCGenSampleShMem
{
    static constexpr int32_t PX = 8;
    static constexpr int32_t PY = 8;

    // PWC Distribution over the shared memory
    using ProductSampler8x8 = ProductSampler<THREAD_PER_BLOCK, X, Y, PX, PY>;
    using BlockDist2D = BlockPWCDistribution2D<THREAD_PER_BLOCK, X, Y>;
    using BlockFilter2D = BlockTextureFilter2D<THREAD_PER_BLOCK, X, Y, GaussFilter>;
    using ConeTracer = BatchConeTracer<THREAD_PER_BLOCK, X, Y>;
    union
    {
        typename BlockDist2D::TempStorage   sDistMem;
        typename ProductSampler8x8::SharedStorage sProductSamplerMem;
        typename BlockFilter2D::TempStorage sFilterMem;
        typename ConeTracer::TempStorage sTraceMem;
    };
    // Bin parameters
    Vector4f sRadianceFieldOriginTMin;
    Vector2f sFieldJitter;
    uint32_t sRayCount;
    uint32_t sOffsetStart;
    uint32_t sNodeId;
    float sBinVoxelSize;
};

template <class RNG>
__device__
inline void CalculateJitterAndBinRayOrigin(Vector4f& posTMin,
                                           Vector2f& jitter,
                                           RNG& rng,
                                           const uint32_t* gRayIds,
                                           const AnisoSVOctreeGPU& svo,
                                           const GPUMetaSurfaceGeneratorGroup& metaSurfGenerator,
                                           Vector2ui rayRange,
                                           uint32_t nodeIdPacked)
{
    // Calculate the voxel size of the bin
    uint32_t nodeId;
    bool isLeaf = ReadSVONodeId(nodeId, nodeIdPacked);
    float binVoxelSize = svo.NodeVoxelSize(nodeId, isLeaf);

    // Utilize voxel center
    Vector3f position = svo.NodePosition(nodeId, isLeaf);

    // Utilize a random ray
    //uint32_t rayCount = rayRange[1] - rayRange[0];
    //uint32_t randomRayIndex = static_cast<uint32_t>(rng.Uniform() * rayCount);
    //uint32_t rayId = gRayIds[rayRange[0] + randomRayIndex];
    //RayReg rayReg = metaSurfGenerator.Ray(rayId);
    //Vector3f position = rayReg.ray.AdvancedPos(rayReg.tMax);

    // TODO: Better offset maybe?
    float tMin = (binVoxelSize * MathConstants::Sqrt3 +
                  MathConstants::LargeEpsilon);
    //float tMin = binVoxelSize + MathConstants::Epsilon;

    // Out
    posTMin = Vector4f(position, tMin);
    jitter = Vector2f(0.5f);// rng.Uniform2D();
}

template <class RNG, int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
__device__
inline void LoadBinInfo(//Output
                        KCGenSampleShMem<THREAD_PER_BLOCK, X, Y>* sharedMem,
                        // I-O
                        RNG& rng,
                        // Input
                        // Per ray
                        const RayId* gRayIds,
                        const GPUMetaSurfaceGeneratorGroup& metaSurfGenerator,
                        // Per bin
                        const uint32_t* gBinOffsets,
                        const uint32_t* gNodeIds,
                        // Constants
                        const AnisoSVOctreeGPU& svo,
                        uint32_t binIndex)
{
    const int32_t THREAD_ID = threadIdx.x;
    const int32_t isMainThread = (THREAD_ID == 0);
    // Load Bin information
    if(isMainThread)
    {
        Vector2ui rayRange = Vector2ui(gBinOffsets[binIndex], gBinOffsets[binIndex + 1]);
        sharedMem->sRayCount = rayRange[1] - rayRange[0];
        sharedMem->sOffsetStart = rayRange[0];
        sharedMem->sNodeId = gNodeIds[binIndex];

        if(sharedMem->sNodeId != INVALID_BIN_ID)
        {
            uint32_t nodeId;
            bool isLeaf = ReadSVONodeId(nodeId, sharedMem->sNodeId);
            sharedMem->sBinVoxelSize = svo.NodeVoxelSize(nodeId, isLeaf);

            // Calculate the voxel size of the bin
            Vector2f jitter;
            Vector4f posTMin;
            CalculateJitterAndBinRayOrigin(posTMin, jitter,
                                           rng,
                                           gRayIds,
                                           svo,
                                           metaSurfGenerator,
                                           rayRange,
                                           sharedMem->sNodeId);
            // Write
            sharedMem->sRadianceFieldOriginTMin = posTMin;
            sharedMem->sFieldJitter = jitter;
        }
    }
    __syncthreads();
}

template <class RNG, class WrapF, class ProjF, int32_t DATA_PER_THREAD,
          int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
__device__
inline void GenerateRadianceField(// Output
                                  float(&filteredRadiances)[DATA_PER_THREAD],
                                  // I-O
                                  RNG& rng,
                                  KCGenSampleShMem<THREAD_PER_BLOCK, X, Y>* sharedMem,
                                  // Constants
                                  const WrapF& WrapFunc,
                                  const ProjF& ProjectionFunc,
                                  const GaussFilter& RFieldGaussFilter,
                                  const AnisoSVOctreeGPU& svo,
                                  float coneAperture)
{
    static constexpr int32_t RT_CONTRIBUTING_THREAD_COUNT = (THREAD_PER_BLOCK < X* Y) ? THREAD_PER_BLOCK : (X * Y);

    const int32_t THREAD_ID = threadIdx.x;

    using SharedMemType = KCGenSampleShMem<THREAD_PER_BLOCK, X, Y>;
    using Filter2D = typename SharedMemType::BlockFilter2D;
    using ConeTracer = typename SharedMemType::ConeTracer;
    static_assert(DATA_PER_THREAD == ConeTracer::DATA_PER_THREAD);

    ConeTracer coneTracer(sharedMem->sTraceMem, svo);

    Vector4f rayOriginTMin = sharedMem->sRadianceFieldOriginTMin;
    float tMin = rayOriginTMin[3];
    Vector3f rayOrigin = Vector3f(rayOriginTMin);

    // Uniform Test
    //for(int i = 0; i < RT_ITER_COUNT; i++)
    //    incRadiances[i] = 1.0f;

    // Batched Cone Trace
    Vector3f    rayDirOut[DATA_PER_THREAD];
    float       tMinOut[DATA_PER_THREAD];
    bool        isLeaf[DATA_PER_THREAD];
    uint32_t    nodeId[DATA_PER_THREAD];
    coneTracer.BatchedConeTraceRay(rayDirOut,
                                   tMinOut,
                                   isLeaf,
                                   nodeId,
                                   // Inputs
                                   rayOrigin,
                                   Vector2f(tMin, FLT_MAX),
                                   coneAperture, 0,
                                   // Functors
                                   ProjectionFunc);

    // Query the radiance from the hit location
    float incRadiances[DATA_PER_THREAD];
    #if 1
    // Non-filtered version
    // Incoming radiance query (result of the trace)
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        // This case occurs only when there is more threads than pixels
        if(THREAD_ID >= RT_CONTRIBUTING_THREAD_COUNT) continue;

        incRadiances[i] = svo.ReadRadiance(rayDirOut[i], coneAperture,
                                           nodeId[i], isLeaf[i]);
        if(incRadiances[i] == 0.0f)
            incRadiances[i] = MathConstants::LargeEpsilon;
    }
    #else
    // Filtered version
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        if(THREAD_ID >= RT_CONTRIBUTING_THREAD_COUNT) continue;

        Vector3f hitPos = (sharedMem->sRadianceFieldOrigin +
                           rayDirOut[i].Normalize() * tMinOut[i]);
        incRadiances[i] = ReadInterpolatedRadiance(hitPos, rayDirOut[i],
                                                   coneAperture,
                                                   svo);

        if(incRadiances[i] == 0.0f)
            incRadiances[i] = MathConstants::LargeEpsilon;
    }
    #endif

    // We finished tracing rays from the scene
    // Now generate distribution from the data
    // and sample for each ray
    // Before that filter the radiance field
    Filter2D(sharedMem->sFilterMem).Filter(filteredRadiances,
                                           incRadiances,
                                           RFieldGaussFilter,
                                           WrapFunc);

    //// Do filtering direct write
    //#pragma unroll
    //for(int i = 0; i < DATA_PER_THREAD; i++)
    //    filteredRadiances[i] = incRadiances[i];

    __syncthreads();
}

// Main directional distribution generation kernel
// Each block is responsible of a single bin
// Each block will trace the SVO to generate
//  omni-directional distribution map.
template <class RNG, int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
__global__ __launch_bounds__(THREAD_PER_BLOCK)
static void KCGenAndSampleDistribution(// Output
                                       RayAuxWFPG* gRayAux,
                                       // I-O
                                       RNGeneratorGPUI** gRNGs,
                                       // Input
                                       // Per-ray
                                       const RayId* gRayIds,
                                       const GPUMetaSurfaceGeneratorGroup metaSurfGenerator,
                                       // Per bin
                                       const uint32_t* gBinOffsets,
                                       const uint32_t* gNodeIds,
                                       // Constants
                                       const GaussFilter RFieldGaussFilter,
                                       float coneAperture,
                                       const AnisoSVOctreeGPU svo,
                                       uint32_t binCount,
                                       bool purePG,
                                       float misRatio)
{
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);
    const int32_t THREAD_ID = threadIdx.x;

    // Number of threads that contributes to the ray tracing operation
    static constexpr int32_t RT_CONTRIBUTING_THREAD_COUNT = (THREAD_PER_BLOCK < X * Y) ? THREAD_PER_BLOCK : (X * Y);
    // How many rows can we process in parallel
    static constexpr int32_t ROW_PER_ITERATION = RT_CONTRIBUTING_THREAD_COUNT / X;
    // How many iterations the entire image would take
    static constexpr int32_t ROW_ITER_COUNT = Y / ROW_PER_ITERATION;
    static_assert(RT_CONTRIBUTING_THREAD_COUNT % X == 0, "RT_THREADS must be multiple of X or vice versa.");
    static_assert(Y % ROW_PER_ITERATION == 0, "RT_THREADS must exactly iterate over X * Y");
    // Ray tracing related
    static constexpr int32_t RT_ITER_COUNT = ROW_ITER_COUNT;

    // PWC Distribution over the shared memory
    using SharedMemType = KCGenSampleShMem<THREAD_PER_BLOCK, X, Y>;
    using Distribution2D = typename SharedMemType::BlockDist2D;
    using ProductSampler8x8 = typename SharedMemType::ProductSampler8x8;
    using Filter2D = typename SharedMemType::BlockFilter2D;
    using ConeTracer = typename SharedMemType::ConeTracer;
    static_assert(RT_ITER_COUNT == ConeTracer::DATA_PER_THREAD);

    // Change the type of the shared memory
    extern __shared__ Byte sharedMemRAW[];
    SharedMemType* sharedMem = reinterpret_cast<SharedMemType*>(sharedMemRAW);

    // Functors for batched cone trace
    auto ProjectionFunc = [&](const Vector2i& localPixelId,
                              const Vector2i& segmentSize)
    {
        Vector2f xi = sharedMem->sFieldJitter;
        Vector2f st = Vector2f(localPixelId) + xi;
        st /= Vector2f(segmentSize);
        Vector3 result = Utility::CocentricOctohedralToDirection(st);
        Vector3 dirYUp = Vector3(result[1], result[2], result[0]);
        return dirYUp;
    };
    auto InvProjectionFunc = [](const Vector3f& direction)
    {
        Vector3 dirZUp = Vector3(direction[2], direction[0], direction[1]);
        return Utility::DirectionToCocentricOctohedral(dirZUp);
    };
    auto NormProjectionFunc = [](const Vector2f& st)
    {
        Vector3f dir = Utility::CocentricOctohedralToDirection(st);
        return Vector3(dir[1], dir[2], dir[0]);
    };

    // Gen wrap functor
    auto WrapFunc = [](const Vector2i& pixelId,
                       const Vector2i& segmentSize)
    {
        return Utility::CocentricOctohedralWrapInt(pixelId, segmentSize);
    };

    ConeTracer coneTracer(sharedMem->sTraceMem, svo);
    // For each block (we allocate enough blocks for the GPU)
    // Each block will process multiple bins
    for(uint32_t binIndex = blockIdx.x; binIndex < binCount;
        binIndex += gridDim.x)
    {
        static constexpr auto LoadBinInfoFunc = LoadBinInfo<RNG, THREAD_PER_BLOCK, X, Y>;
        // Load the bin Info
        LoadBinInfoFunc(sharedMem, rng, gRayIds, metaSurfGenerator,
                        gBinOffsets, gNodeIds, svo, binIndex);

        //// Kill the entire block if Node Id is invalid
        if(sharedMem->sNodeId == INVALID_BIN_ID)
        {
            __syncthreads();
            continue;
        }

        // Generate Radiance Field
        float filteredRadiances[RT_ITER_COUNT];
        static constexpr auto GenRadFieldFunc = GenerateRadianceField<RNG,
                                                                      decltype(WrapFunc),
                                                                      decltype(ProjectionFunc),
                                                                      RT_ITER_COUNT,
                                                                      THREAD_PER_BLOCK, X, Y>;
        GenRadFieldFunc(filteredRadiances,
                        rng,
                        sharedMem,
                        WrapFunc,
                        ProjectionFunc,
                        RFieldGaussFilter,
                        svo,
                        coneAperture);

        // Non product sample version
        // Generate PWC Distribution over the radiances
        Distribution2D dist2D(sharedMem->sDistMem, filteredRadiances);
        // Block threads will loop over the every ray in this bin
        for(uint32_t rayIndex = THREAD_ID; rayIndex < sharedMem->sRayCount;
            rayIndex += THREAD_PER_BLOCK)
        {
            // Let the ray acquire the surface
            uint32_t rayId = gRayIds[sharedMem->sOffsetStart + rayIndex];
            GPUMetaSurface surf = metaSurfGenerator.AcquireWork(rayId);

            float sampleRatio = (purePG) ? 1.0f : misRatio;
            float xi = rng.Uniform();
            Vector3f wi = -(metaSurfGenerator.Ray(rayId).ray.getDirection());

            Vector2f sampledUV;
            float pdfSampled, pdfOther;
            if(xi >= sampleRatio)
            {
                // Sample Using BxDF
                RayF wo;
                const GPUMediumI* outMedium;
                surf.Sample(wo, pdfSampled,
                            outMedium,
                            //
                            wi,
                            GPUMediumVacuum(0),
                            rng);
                sampledUV = InvProjectionFunc(wo.getDirection());

                pdfOther = dist2D.Pdf(sampledUV * Vector2f(X, Y));
                pdfOther *= 0.25f * MathConstants::InvPi;
                //pdfOther *= 2;
                sampleRatio = 1.0f - sampleRatio;
            }
            else
            {
                // Sample Using Radiance Field
                Vector2f index;
                sampledUV = dist2D.Sample(pdfSampled, index, rng);
                pdfSampled *= 0.25f * MathConstants::InvPi;
                //pdfSampled *= 2;

                Vector3f wo = NormProjectionFunc(sampledUV);
                pdfOther = surf.Pdf(wo, wi, GPUMediumVacuum(0));
            }
            // MIS
            using namespace TracerFunctions;
            float pdf = (pdfSampled * sampleRatio /
                         BalanceHeuristic(sampleRatio, pdfSampled,
                                          1.0f - sampleRatio, pdfOther));
            pdf = (pdfSampled == 0.0f) ? 0.0f : pdf;

            gRayAux[rayId].guideDir = Vector2h(sampledUV[0], sampledUV[1]);
            gRayAux[rayId].guidePDF = pdf;
        }

        // Wait all to finish
        __syncthreads();
    }
}

template <class RNG, int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
__global__ __launch_bounds__(THREAD_PER_BLOCK)
static void KCGenAndSampleDistributionProduct(// Output
                                              RayAuxWFPG* gRayAux,
                                              // I-O
                                              RNGeneratorGPUI** gRNGs,
                                              // Input
                                              // Per-ray
                                              const RayId* gRayIds,
                                              const GPUMetaSurfaceGeneratorGroup metaSurfGenerator,
                                              // Per bin
                                              const uint32_t* gBinOffsets,
                                              const uint32_t* gNodeIds,
                                              // Constants
                                              const GaussFilter RFieldGaussFilter,
                                              float coneAperture,
                                              const AnisoSVOctreeGPU svo,
                                              uint32_t binCount,
                                              bool purePG,
                                              float misRatio)
{
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);
    const int32_t THREAD_ID = threadIdx.x;

    // Number of threads that contributes to the ray tracing operation
    static constexpr int32_t RT_CONTRIBUTING_THREAD_COUNT = (THREAD_PER_BLOCK < X * Y) ? THREAD_PER_BLOCK : (X * Y);
    // How many rows can we process in parallel
    static constexpr int32_t ROW_PER_ITERATION = RT_CONTRIBUTING_THREAD_COUNT / X;
    // How many iterations the entire image would take
    static constexpr int32_t ROW_ITER_COUNT = Y / ROW_PER_ITERATION;
    static_assert(RT_CONTRIBUTING_THREAD_COUNT % X == 0, "RT_THREADS must be multiple of X or vice versa.");
    static_assert(Y % ROW_PER_ITERATION == 0, "RT_THREADS must exactly iterate over X * Y");
    // Ray tracing related
    static constexpr int32_t RT_ITER_COUNT = ROW_ITER_COUNT;

    // PWC Distribution over the shared memory
    using SharedMemType = KCGenSampleShMem<THREAD_PER_BLOCK, X, Y>;
    using ProductSampler8x8 = typename SharedMemType::ProductSampler8x8;

    // Change the type of the shared memory
    extern __shared__ Byte sharedMemRAW[];
    SharedMemType* sharedMem = reinterpret_cast<SharedMemType*>(sharedMemRAW);

    // Functors for batched cone trace
    auto ProjectionFunc = [&](const Vector2i& localPixelId,
                              const Vector2i& segmentSize)
    {
        // Jitter the values
        Vector2f xi = sharedMem->sFieldJitter;
        Vector2f st = Vector2f(localPixelId) + xi;
        st /= Vector2f(segmentSize);
        Vector3 result = Utility::CocentricOctohedralToDirection(st);
        Vector3 dirYUp = Vector3(result[1], result[2], result[0]);
        return dirYUp;
    };
    auto InvProjectionFunc = [](const Vector3f& direction)
    {
        Vector3 dirZUp = Vector3(direction[2], direction[0], direction[1]);
        return Utility::DirectionToCocentricOctohedral(dirZUp);
    };
    auto NormProjectionFunc = [](const Vector2f& st)
    {
        Vector3f dir = Utility::CocentricOctohedralToDirection(st);
        return Vector3(dir[1], dir[2], dir[0]);
    };
    auto WrapFunc = [](const Vector2i& pixelId,
                       const Vector2i& segmentSize)
    {
        return Utility::CocentricOctohedralWrapInt(pixelId, segmentSize);
    };

    static constexpr auto GenRadFieldFunc = GenerateRadianceField<RNG,
                                                                  decltype(WrapFunc),
                                                                  decltype(ProjectionFunc),
                                                                  RT_ITER_COUNT,
                                                                  THREAD_PER_BLOCK, X, Y>;
    static constexpr auto LoadBinInfoFunc = LoadBinInfo<RNG, THREAD_PER_BLOCK, X, Y>;

    // For each block (we allocate enough blocks for the GPU)
    // Each block will process multiple bins
    for(uint32_t binIndex = blockIdx.x; binIndex < binCount;
        binIndex += gridDim.x)
    {
        // Load the bin Info
        LoadBinInfoFunc(sharedMem, rng,
                        gRayIds, metaSurfGenerator,
                        gBinOffsets, gNodeIds,
                        svo, binIndex);
        // Kill the entire block if Node Id is invalid
        if(sharedMem->sNodeId == INVALID_BIN_ID)
        {
            __syncthreads();
            continue;
        }

        // Generate Radiance Field
        float filteredRadiances[RT_ITER_COUNT];
        GenRadFieldFunc(filteredRadiances,
                        rng,
                        sharedMem,
                        WrapFunc,
                        ProjectionFunc,
                        RFieldGaussFilter,
                        svo,
                        coneAperture);

        static constexpr uint32_t WARP_PER_BLOCK = THREAD_PER_BLOCK / WARP_SIZE;
        // Parallelization logic changes now it is one ray per warp
        // Generate the product sampler first
        // Parallelization logic internally is different, block threads
        // collaboratively compiles a outer 8x8 radiance field and each
        // of these have inner (N/8)x(M/8) radiance field
        ProductSampler8x8 productSampler(sharedMem->sProductSamplerMem,
                                         filteredRadiances,
                                         gRayIds + sharedMem->sOffsetStart,
                                         metaSurfGenerator);
        // Block warp-stride loop
        const uint32_t warpId = THREAD_ID / WARP_SIZE;
        const bool isWarpLeader = (THREAD_ID % WARP_SIZE) == 0;
        // For each ray
        for(uint32_t rayIndex = warpId; rayIndex < sharedMem->sRayCount;
            rayIndex += WARP_PER_BLOCK)
        {
            float pdf;
            Vector2f uv;
            if(purePG)
            {
                // Sample using the surface/material,
                // fetch the rays surface from metaSurfaceGenerator,
                // multiply the 8x8 outer field with the evaluated material,
                // create 8x8 PWC CDF for rows and the column. Find the region
                // then on the inner region do couple of 2-3 round of rejection sampling
                // (since a warp is responsible for a ray, each round calculates 32 samples)
                // Then multiply the PDFs for inner and outer etc. and return the sample
                uv = productSampler.SampleWithProduct(pdf, rng,
                                                      rayIndex,
                                                      ProjectionFunc);
                // Our projection function is co-centric octahedral so it is area preserving
                // directly divide with Omega (4 * PI)
                pdf *= 0.25f * MathConstants::InvPi;
            }
            else
            {
                // Same as above but also combine with MIS (BxDF <=> Guide)
                uv = productSampler.SampleMIS(pdf,
                                              rng,
                                              rayIndex,
                                              ProjectionFunc,
                                              InvProjectionFunc,
                                              NormProjectionFunc,
                                              misRatio,
                                              0.25f * MathConstants::InvPi);
            }

            // Only warp leader has valid values
            if(isWarpLeader)
            {
                if(isnan(pdf) || uv.HasNaN())
                {
                    printf("NaN uv(%f, %f), pdf(%f)\n", uv[0], uv[1], pdf);
                    uv = Vector2f(0.5f);
                    pdf = 1.0f;
                }
                // Store the sampled direction of the ray
                uint32_t rayId = gRayIds[sharedMem->sOffsetStart + rayIndex];
                gRayAux[rayId].guideDir = Vector2h(uv[0], uv[1]);
                gRayAux[rayId].guidePDF = pdf;
            }
        }

        // Wait all to finish
        __syncthreads();
    }
}

template <class RNG, int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
__global__ __launch_bounds__(THREAD_PER_BLOCK)
static void KCSampleDistributionOptiX(// Output
                                      RayAuxWFPG* gRayAux,
                                      // I-O
                                      RNGeneratorGPUI** gRNGs,
                                      // Input
                                      // Per-ray
                                      const RayId* gRayIds,
                                      const GPUMetaSurfaceGeneratorGroup metaSurfGenerator,
                                      // Per bin
                                      const uint32_t* gBinOffsets,
                                      const uint32_t* gNodeIds,
                                      // Buffer
                                      SVOOptixRadianceBuffer::SegmentedField<const float*> radBuffer,
                                      // Constants
                                      const GaussFilter RFieldGaussFilter,
                                      const AnisoSVOctreeGPU svo,
                                      uint32_t binCount,
                                      bool purePG,
                                      float misRatio)
{
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);
    const int32_t THREAD_ID = threadIdx.x;
    const bool isMainThread = (THREAD_ID == 0);
    // Number of threads that contributes to the ray tracing operation
    static constexpr int32_t RT_CONTRIBUTING_THREAD_COUNT = (THREAD_PER_BLOCK < X * Y) ? THREAD_PER_BLOCK : (X * Y);
    // How many rows can we process in parallel
    static constexpr int32_t ROW_PER_ITERATION = RT_CONTRIBUTING_THREAD_COUNT / X;
    // How many iterations the entire image would take
    static constexpr int32_t ROW_ITER_COUNT = Y / ROW_PER_ITERATION;
    static_assert(RT_CONTRIBUTING_THREAD_COUNT % X == 0, "RT_THREADS must be multiple of X or vice versa.");
    static_assert(Y % ROW_PER_ITERATION == 0, "RT_THREADS must exactly iterate over X * Y");
    // Ray tracing related
    static constexpr int32_t RT_ITER_COUNT = ROW_ITER_COUNT;

    // PWC Distribution over the shared memory
    using SharedMemType = KCGenSampleShMem<THREAD_PER_BLOCK, X, Y>;
    using Distribution2D = typename SharedMemType::BlockDist2D;
    using Filter2D = typename SharedMemType::BlockFilter2D;

    // Functors for batched cone trace
    auto InvProjectionFunc = [](const Vector3f& direction)
    {
        Vector3 dirZUp = Vector3(direction[2], direction[0], direction[1]);
        return Utility::DirectionToCocentricOctohedral(dirZUp);
    };
    auto NormProjectionFunc = [](const Vector2f& st)
    {
        Vector3f dir = Utility::CocentricOctohedralToDirection(st);
        return Vector3(dir[1], dir[2], dir[0]);
    };

    // Gen wrap functor
    auto WrapFunc = [](const Vector2i& pixelId,
                       const Vector2i& segmentSize)
    {
        return Utility::CocentricOctohedralWrapInt(pixelId, segmentSize);
    };

    // Change the type of the shared memory
    extern __shared__ Byte sharedMemRAW[];
    SharedMemType* sharedMem = reinterpret_cast<SharedMemType*>(sharedMemRAW);

    // For each block (we allocate enough blocks for the GPU)
    // Each block will process multiple bins
    for(uint32_t binIndex = blockIdx.x; binIndex < binCount;
        binIndex += gridDim.x)
    {
        // Load the bin Info
        if(isMainThread)
        {
            Vector2ui rayRange = Vector2ui(gBinOffsets[binIndex], gBinOffsets[binIndex + 1]);
            sharedMem->sRayCount = rayRange[1] - rayRange[0];
            sharedMem->sOffsetStart = rayRange[0];
            sharedMem->sNodeId = gNodeIds[binIndex];
        }
        __syncthreads();

        // Kill the entire block if Node Id is invalid
        if(sharedMem->sNodeId == INVALID_BIN_ID)
        {
            __syncthreads();
            continue;
        }

        // Load Radiance Field
        float incRadiances[RT_ITER_COUNT];
        for(int i = 0; i < RT_ITER_COUNT; i++)
        {
            int32_t localId = i * THREAD_PER_BLOCK + THREAD_ID;
            if(localId >= X * Y) continue;

            incRadiances[i] = radBuffer[binIndex][localId];
        };

        // Generate Radiance Field
        float filteredRadiances[RT_ITER_COUNT];
        Filter2D(sharedMem->sFilterMem).Filter(filteredRadiances,
                                               incRadiances,
                                               RFieldGaussFilter,
                                               WrapFunc);

        // Non product sample version
        // Generate PWC Distribution over the radiances
        Distribution2D dist2D(sharedMem->sDistMem, filteredRadiances);
        // Block threads will loop over the every ray in this bin
        for(uint32_t rayIndex = THREAD_ID; rayIndex < sharedMem->sRayCount;
            rayIndex += THREAD_PER_BLOCK)
        {
            // Let the ray acquire the surface
            uint32_t rayId = gRayIds[sharedMem->sOffsetStart + rayIndex];
            GPUMetaSurface surf = metaSurfGenerator.AcquireWork(rayId);

            float sampleRatio = (purePG) ? 1.0f : misRatio;
            float xi = rng.Uniform();
            Vector3f wi = -(metaSurfGenerator.Ray(rayId).ray.getDirection());

            Vector2f sampledUV;
            float pdfSampled, pdfOther;
            if(xi >= sampleRatio)
            {
                // Sample Using BxDF
                RayF wo;
                const GPUMediumI* outMedium;
                surf.Sample(wo, pdfSampled,
                            outMedium,
                            //
                            wi,
                            GPUMediumVacuum(0),
                            rng);
                sampledUV = InvProjectionFunc(wo.getDirection());

                pdfOther = dist2D.Pdf(sampledUV * Vector2f(X, Y));
                pdfOther *= 0.25f * MathConstants::InvPi;
                //pdfOther *= 2;
                sampleRatio = 1.0f - sampleRatio;
            }
            else
            {
                // Sample Using Radiance Field
                Vector2f index;
                sampledUV = dist2D.Sample(pdfSampled, index, rng);
                pdfSampled *= 0.25f * MathConstants::InvPi;
                //pdfSampled *= 2;

                Vector3f wo = NormProjectionFunc(sampledUV);
                pdfOther = surf.Pdf(wo, wi, GPUMediumVacuum(0));
            }
            // MIS
            using namespace TracerFunctions;
            float pdf = (pdfSampled * sampleRatio /
                         BalanceHeuristic(sampleRatio, pdfSampled,
                                          1.0f - sampleRatio, pdfOther));
            pdf = (pdfSampled == 0.0f) ? 0.0f : pdf;

            gRayAux[rayId].guideDir = Vector2h(sampledUV[0], sampledUV[1]);
            gRayAux[rayId].guidePDF = pdf;
        }

        // Wait all to finish
        __syncthreads();
    }
}

template <class RNG, int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
__global__ __launch_bounds__(THREAD_PER_BLOCK)
static void KCSampleDistributionProductOptiX(// Output
                                             RayAuxWFPG* gRayAux,
                                             // I-O
                                             RNGeneratorGPUI** gRNGs,
                                             // Input
                                             // Per-ray
                                             const RayId* gRayIds,
                                             const GPUMetaSurfaceGeneratorGroup metaSurfGenerator,
                                             // Per bin
                                             const uint32_t* gBinOffsets,
                                             const uint32_t* gNodeIds,
                                             // Buffer
                                             SVOOptixRadianceBuffer::SegmentedField<const float*> radBuffer,
                                             // Constants
                                             const GaussFilter RFieldGaussFilter,
                                             const AnisoSVOctreeGPU svo,
                                             uint32_t binCount,
                                             bool purePG,
                                             float misRatio)
{
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);
    const int32_t THREAD_ID = threadIdx.x;
    const bool isMainThread = (THREAD_ID == 0);
    // Number of threads that contributes to the ray tracing operation
    static constexpr int32_t RT_CONTRIBUTING_THREAD_COUNT = (THREAD_PER_BLOCK < X * Y) ? THREAD_PER_BLOCK : (X * Y);
    // How many rows can we process in parallel
    static constexpr int32_t ROW_PER_ITERATION = RT_CONTRIBUTING_THREAD_COUNT / X;
    // How many iterations the entire image would take
    static constexpr int32_t ROW_ITER_COUNT = Y / ROW_PER_ITERATION;
    static_assert(RT_CONTRIBUTING_THREAD_COUNT % X == 0, "RT_THREADS must be multiple of X or vice versa.");
    static_assert(Y % ROW_PER_ITERATION == 0, "RT_THREADS must exactly iterate over X * Y");
    // Ray tracing related
    static constexpr int32_t RT_ITER_COUNT = ROW_ITER_COUNT;

    // PWC Distribution over the shared memory
    using SharedMemType = KCGenSampleShMem<THREAD_PER_BLOCK, X, Y>;
    using ProductSampler8x8 = typename SharedMemType::ProductSampler8x8;
    using Filter2D = typename SharedMemType::BlockFilter2D;

    // Functors for batched cone trace
    auto ProjectionFunc = [&](const Vector2i& localPixelId,
                              const Vector2i& segmentSize)
    {
        // Jitter the values
        //Vector2f xi = Vector2f(0.5f);
        Vector2f xi = rng.Uniform2D();
        Vector2f st = Vector2f(localPixelId) + xi;
        st /= Vector2f(segmentSize);
        Vector3 result = Utility::CocentricOctohedralToDirection(st);
        Vector3 dirYUp = Vector3(result[1], result[2], result[0]);
        return dirYUp;
    };
    auto InvProjectionFunc = [](const Vector3f& direction)
    {
        Vector3 dirZUp = Vector3(direction[2], direction[0], direction[1]);
        return Utility::DirectionToCocentricOctohedral(dirZUp);
    };
    auto NormProjectionFunc = [](const Vector2f& st)
    {
        Vector3f dir = Utility::CocentricOctohedralToDirection(st);
        return Vector3(dir[1], dir[2], dir[0]);
    };
    auto WrapFunc = [](const Vector2i& pixelId,
                       const Vector2i& segmentSize)
    {
        return Utility::CocentricOctohedralWrapInt(pixelId, segmentSize);
    };

    // Change the type of the shared memory
    extern __shared__ Byte sharedMemRAW[];
    SharedMemType* sharedMem = reinterpret_cast<SharedMemType*>(sharedMemRAW);

    // For each block (we allocate enough blocks for the GPU)
    // Each block will process multiple bins
    for(uint32_t binIndex = blockIdx.x; binIndex < binCount;
        binIndex += gridDim.x)
    {
        // Load the bin Info
        if(isMainThread)
        {
            Vector2ui rayRange = Vector2ui(gBinOffsets[binIndex], gBinOffsets[binIndex + 1]);
            sharedMem->sRayCount = rayRange[1] - rayRange[0];
            sharedMem->sOffsetStart = rayRange[0];
            sharedMem->sNodeId = gNodeIds[binIndex];
        }
        __syncthreads();

        // Kill the entire block if Node Id is invalid
        if(sharedMem->sNodeId == INVALID_BIN_ID)
        {
            __syncthreads();
            continue;
        }

        // Load Radiance Field
        float incRadiances[RT_ITER_COUNT];
        for(int i = 0; i < RT_ITER_COUNT; i++)
        {
            int32_t localId = i * THREAD_PER_BLOCK + THREAD_ID;
            if(localId >= X * Y) continue;

            incRadiances[i] = radBuffer[binIndex][localId];
        };

        // Generate Radiance Field
        float filteredRadiances[RT_ITER_COUNT];
        Filter2D(sharedMem->sFilterMem).Filter(filteredRadiances,
                                               incRadiances,
                                               RFieldGaussFilter,
                                               WrapFunc);

        static constexpr uint32_t WARP_PER_BLOCK = THREAD_PER_BLOCK / WARP_SIZE;
        // Parallelization logic changes now it is one ray per warp
        // Generate the product sampler first
        // Parallelization logic internally is different, block threads
        // collaboratively compiles a outer 8x8 radiance field and each
        // of these have inner (N/8)x(M/8) radiance field
        ProductSampler8x8 productSampler(sharedMem->sProductSamplerMem,
                                         filteredRadiances,
                                         gRayIds + sharedMem->sOffsetStart,
                                         metaSurfGenerator);
        // Block warp-stride loop
        const uint32_t warpId = THREAD_ID / WARP_SIZE;
        const bool isWarpLeader = (THREAD_ID % WARP_SIZE) == 0;
        // For each ray
        for(uint32_t rayIndex = warpId; rayIndex < sharedMem->sRayCount;
            rayIndex += WARP_PER_BLOCK)
        {
            float pdf;
            Vector2f uv;
            if(purePG)
            {
                // Sample using the surface/material,
                // fetch the rays surface from metaSurfaceGenerator,
                // multiply the 8x8 outer field with the evaluated material,
                // create 8x8 PWC CDF for rows and the column. Find the region
                // then on the inner region do couple of 2-3 round of rejection sampling
                // (since a warp is responsible for a ray, each round calculates 32 samples)
                // Then multiply the PDFs for inner and outer etc. and return the sample
                uv = productSampler.SampleWithProduct(pdf, rng,
                                                      rayIndex,
                                                      ProjectionFunc);
                // Our projection function is co-centric octahedral so it is area preserving
                // directly divide with Omega (4 * PI)
                pdf *= 0.25f * MathConstants::InvPi;
            }
            else
            {
                // Same as above but also combine with MIS (BxDF <=> Guide)
                uv = productSampler.SampleMIS(pdf,
                                              rng,
                                              rayIndex,
                                              ProjectionFunc,
                                              InvProjectionFunc,
                                              NormProjectionFunc,
                                              misRatio,
                                              0.25f * MathConstants::InvPi);
            }

            // Only warp leader has valid values
            if(isWarpLeader)
            {
                if(isnan(pdf) || uv.HasNaN())
                {
                    printf("NaN uv(%f, %f), pdf(%f)\n", uv[0], uv[1], pdf);
                    uv = Vector2f(0.5f);
                    pdf = 1.0f;
                }
                // Store the sampled direction of the ray
                uint32_t rayId = gRayIds[sharedMem->sOffsetStart + rayIndex];
                gRayAux[rayId].guideDir = Vector2h(uv[0], uv[1]);
                gRayAux[rayId].guidePDF = pdf;
            }
        }

        // Wait all to finish
        __syncthreads();
    }
}

template <class RNG>
__global__
static void KCGenerateBinInfoOptiX(// Output
                                   Vector4f* dRadianceFieldRayOrigins,
                                   Vector2f* dProjectionJitters,
                                   // I-O
                                   RNGeneratorGPUI** gRNGs,
                                   // Input
                                   const RayId* gRayIds,
                                   const GPUMetaSurfaceGeneratorGroup metaSurfGenerator,
                                   // Per bin
                                   const uint32_t* gBinOffsets,
                                   const uint32_t* gNodeIds,
                                   // Constants
                                   const AnisoSVOctreeGPU svo,
                                   uint32_t partitionCount)
{
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);

    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < partitionCount;
        threadId += (blockDim.x * gridDim.x))
    {
        Vector2ui rayRange = Vector2ui(gBinOffsets[threadId], gBinOffsets[threadId + 1]);
        uint32_t nodeIdPacked = gNodeIds[threadId];

        if(nodeIdPacked != INVALID_BIN_ID)
        {
            Vector2f jitter;
            Vector4f posTMin;
            CalculateJitterAndBinRayOrigin(posTMin, jitter,
                                           rng,
                                           gRayIds,
                                           svo,
                                           metaSurfGenerator,
                                           rayRange,
                                           nodeIdPacked);

            dRadianceFieldRayOrigins[threadId] = posTMin;
            dProjectionJitters[threadId] = jitter;
        }
        else
        {
            dRadianceFieldRayOrigins[threadId] = Vector4f(NAN);
            dProjectionJitters[threadId] = Vector2f(NAN);
        }
    }
}