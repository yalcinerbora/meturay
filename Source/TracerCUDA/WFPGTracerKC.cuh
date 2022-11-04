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

#include "TracerFunctions.cuh"
#include "TracerConstants.h"

#include "RayLib/RandomColor.h"

#include "AnisoSVO.cuh"

#include "GPUBlockPWCDistribution.cuh"
#include "GPUBlockPWLDistribution.cuh"
#include "BlockTextureFilter.cuh"

#include "GPUCameraPixel.cuh"
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
        const float BxDF_GuideSampleRatio = (renderState.skipPG) ? 0.0f : 1.0f;
        float xi = rng.Uniform();

        bool selectedPDFZero = false;
        float pdfBxDF, pdfGuide;
        if(xi >= BxDF_GuideSampleRatio)
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
            pdfGuide = aux.guidePDF;

            selectedPDFZero = (pdfBxDF == 0.0f);
        }
        else
        {
            // Sample a path from the pre-sampled UV
            // uv coordinates to spherical coordinates
            Vector2f uv = Vector2f(aux.guideDir[0], aux.guideDir[1]);
            Vector3f dirZUp = Utility::CocentricOctohedralToDirection(uv);
            Vector3f direction = Vector3f(dirZUp[1], dirZUp[2], dirZUp[0]);
            pdfGuide = aux.guidePDF;

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
            rayPath.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);

            selectedPDFZero = (pdfGuide == 0.0f);
        }
        // Pdf Average
        //pdfPath = pdfBxDF;
        pdfPath = BxDF_GuideSampleRatio          * pdfGuide +
                  (1.0f - BxDF_GuideSampleRatio) * pdfBxDF;
        pdfPath = selectedPDFZero ? 0.0f : pdfPath;

        // DEBUG
        if(isnan(pdfPath) || isnan(pdfBxDF) || isnan(pdfGuide))
            printf("[%s] NAN PDF = % f = w * %f + (1.0f - w) * %f, w: % f\n",
                   (xi >= BxDF_GuideSampleRatio) ? "BxDF": "SVO",
                   pdfPath, pdfBxDF, pdfGuide, BxDF_GuideSampleRatio);
        if(pdfPath != 0.0f && rayPath.getDirection().HasNaN())
            printf("[%s] NAN DIR %f, %f, %f\n",
                    (xi >= BxDF_GuideSampleRatio) ? "BxDF" : "SVO",
                    rayPath.getDirection()[0],
                    rayPath.getDirection()[1],
                    rayPath.getDirection()[2]);
        if(reflectance.HasNaN())
            printf("[%s] NAN REFL %f %f %f\n",
                   (xi >= BxDF_GuideSampleRatio) ? "BxDF" : "SVO",
                   reflectance[0],
                   reflectance[1],
                   reflectance[2]);

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

    auto GenConeAperture = [](const GPUCameraI& gCamera,
                              const Vector2i& resolution) -> float
    {
        Vector2f fov = gCamera.FoV();
        float coneAngleX = fov[0] / static_cast<float>(resolution[0]);
        float coneAngleY = fov[1] / static_cast<float>(resolution[1]);
        // Return average
        return (coneAngleX + coneAngleY) * 0.5f;
    };

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
            shMem->sConeAperture = GenConeAperture(*gCamera, totalPixelCount);
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
        auto WrapFunc = [](const Vector2i& pixelId,
                           const Vector2i& segmentSize)
        {
            return pixelId.Clamp(Vector2i(0), segmentSize - 1);
        };

        float tMin[ConeTracer::DATA_PER_THREAD];
        bool  isLeaf[ConeTracer::DATA_PER_THREAD];
        uint32_t nodeIndex[ConeTracer::DATA_PER_THREAD];
        Vector3f rayDir[ConeTracer::DATA_PER_THREAD];

        batchedConeTracer.RecursiveConeTraceRay(rayDir, tMin, isLeaf,
                                                nodeIndex,
                                                // Inputs
                                                shMem->sPosition,
                                                shMem->sTMinMax,
                                                shMem->sConeAperture,
                                                0, 1, maxQueryLevelOffset,
                                                ProjectionFunc,
                                                WrapFunc);

        // Write as color
        for(int i = 0; i < ConeTracer::DATA_PER_THREAD; i++)
        {
            Vector4f locColor = Vector4f(0.0f, 0.0f, 10.0f, 1.0f);
            // Octree Display Mode
            if(mode == WFPGRenderMode::SVO_FALSE_COLOR)
                locColor = (nodeIndex[i] != UINT32_MAX) ? Vector4f(Utility::RandomColorRGB(nodeIndex[i]), 1.0f)
                                                        : Vector4f(Vector3f(0.0f), 1.0f);
            // Payload Display Mode
            else if(nodeIndex[i] == UINT32_MAX)
                locColor = Vector4f(1.0f, 0.0f, 1.0f, 1.0f);
            else if(mode == WFPGRenderMode::SVO_RADIANCE)
            {
                Vector3f hitPos = shMem->sPosition + rayDir[i].Normalize() * tMin[i];
                float radianceF = ReadInterpolatedRadiance(hitPos, rayDir[i],
                                                           shMem->sConeAperture,
                                                           svo);

                //half radiance = svo.ReadRadiance(rayDir[i], shMem->sConeAperture,
                //                                 nodeIndex[i], isLeaf[i]);
                //float radianceF = radiance;
                //if(radiance != static_cast<half>(MRAY_HALF_MAX))
                    locColor = Vector4f(Vector3f(radianceF), 1.0f);
            }
            else if(mode == WFPGRenderMode::SVO_NORMAL)
            {
                float stdDev;
                Vector3f normal = svo.DebugReadNormal(stdDev, nodeIndex[i], isLeaf[i]);

                // Voxels are two sided show the normal for the current direction
                normal = (normal.Dot(rayDir[i]) >= 0.0f) ? normal : -normal;

                // Convert normal to 0-1 range
                //normal += Vector3f(1.0f);
                //normal *= Vector3f(0.5f);
                locColor = Vector4f(normal, stdDev);
            }

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

    //assert(dirXY < dimensions);
    //using namespace MathConstants;
    //// Spherical coordinate deltas
    //Vector2f deltaXY = Vector2f((2.0f * Pi) / static_cast<float>(dimensions[0]),
    //                            Pi / static_cast<float>(dimensions[1]));

    //// Assume image space bottom left is (0,0)
    //// Center to the pixel as well
    //// Offset
    //Vector2f xi = rng.Uniform2D();
    ////Vector2f xi = Vector2f(0.5f);

    //Vector2f dirXYFloat = Vector2f(dirXY[0], dirXY[1]) + xi;
    //Vector2f sphrCoords = Vector2f(-Pi + dirXYFloat[0] * deltaXY[0],
    //                               Pi - dirXYFloat[1] * deltaXY[1]);
    //Vector3f result = Utility::SphericalToCartesianUnit(sphrCoords);
    //// Spherical Coords calculates as Z up change it to Y up
    //Vector3 dirYUp = Vector3(result[1], result[2], result[0]);

    ////printf("Pixel [%u, %u], ThetaPhi [%f, %f], Dir[%f, %f, %f]\n",
    ////       dirXY[0], dirXY[1],
    ////       sphrCoords[0] * RadToDegCoef,
    ////       sphrCoords[1] * RadToDegCoef,
    ////       dirYUp[0], dirYUp[1], dirYUp[2]);

    return dirYUp;
}

__global__
static void KCInitializeSVOBins(// Outputs
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
            gRayAux[threadId].binId = GenerateLeafIndex(leafIndex);
            // Increment the ray count on that leaf
            svo.IncrementLeafRayCount(leafIndex);
        }
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

template <uint32_t TPB, uint32_t X, uint32_t Y,
          uint32_t PX, uint32_t PY>
class ProductSampler
{
    public:
    static constexpr Vector2ui PRODUCT_MAP_SIZE = Vecto2ui(PX, PY);
    static constexpr uint32_t WARP_PER_BLOCK = TPB / WARP_SIZE;
    static_assert(TPB % WARP_SIZE == 0);

    struct SharedStorage
    {
        // Scratch pad for each warp
        //float sLocalProducts[WARP_PER_BLOCK][PRODUCT_MAP_SIZE[0]][PRODUCT_MAP_SIZE[1]];
        // Main radiance field
        float sRadianceField[X][Y];
        // Reduced radiance field (will be multiplied by the BxDF field)
        float sRadianceFieldSmall[PRODUCT_MAP_SIZE[0]][PRODUCT_MAP_SIZE[1]];
        //
        GPUMetaSurface sSurfaces[WARP_PER_BLOCK];
    };

    private:
    SharedStorage&                          shMem;

    const RayId*                            gRayIds;
    const GPUMetaSurfaceGeneratorGroup&     metaSurfGenerator;
    uint32_t                                rayCount;
    bool                                    isWarpLeader;

    uint32_t                                warpLocalId;
    uint32_t                                warpId;

    public:
    __device__ ProductSampler(SharedStorage& sharedMem,
                              const RayId* asd,
                              const GPUMetaSurfaceGeneratorGroup& def)
        : shMem(sharedMem)
        , gRayIds(asd)
        , metaSurfGenerator(def)
    {}

    __device__
    Vector2f SampleProductMaterial(float& pdf,
                                   RNGeneratorGPUI& rng) const
    {
        // Load material..
        if(isWarpLeader)
            shMem.sSurfaces[warpId] = metaSurfGenerator.AcquireWork(rayId);

        // No need to sync here used threads are in lockstep
        float test[2];

        Vector3f a = shMem.sSurfaces[warpId].Evaluate(Vector3f(1.0f),
                                                      Vector3f(1.0f),
                                                      medium,
                                                      shMem.sSurfaces[warpId].WorldPosition());
        test[0] = Utility::RGBToLuminance(a);
    }

};

// Shared Memory Class of the kernel below
template <uint32_t THREAD_PER_BLOCK, uint32_t X, uint32_t Y>
struct KCGenSampleShMem
{


    // PWC Distribution over the shared memory
    using ProductSampler8x8 = ProductSampler<TPB, X, Y, 8, 8>;
    //using BlockDist2D = BlockPWCDistribution2D<THREAD_PER_BLOCK, X, Y>;
    //using BlockDist2D = BlockPWLDistribution2D<THREAD_PER_BLOCK, X, Y>;
    using BlockFilter2D = BlockTextureFilter2D<GaussFilter, THREAD_PER_BLOCK, X, Y>;
    using ConeTracer = BatchConeTracer<THREAD_PER_BLOCK, X, Y>;
    union
    {
        //typename BlockDist2D::TempStorage   sDistMem;

        typename ProductSampler8x8 sProductSamplerMem;


        typename BlockFilter2D::TempStorage sFilterMem;
        typename ConeTracer::TempStorage sTraceMem;
    };
    // Bin parameters
    Vector3f sRadianceFieldOrigin;
    uint32_t sRayCount;
    uint32_t sOffsetStart;
    uint32_t sNodeId;
    float sBinVoxelSize;
};

// Main directional distribution generation kernel
// Each block is responsible of a single bin
// Each block will trace the SVO to generate
//  omni-directional distribution map.
template <class RNG, uint32_t THREAD_PER_BLOCK, uint32_t X, uint32_t Y>
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
                                       const GaussFilter rFieldGaussFilter,
                                       float coneAperture,
                                       const AnisoSVOctreeGPU svo,
                                       uint32_t binCount)
{
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);
    const uint32_t THREAD_ID = threadIdx.x;
    const uint32_t isMainThread = (THREAD_ID == 0);

    // Number of threads that contributes to the ray tracing operation
    static constexpr uint32_t RT_CONTRIBUTING_THREAD_COUNT = (THREAD_PER_BLOCK < X * Y) ? THREAD_PER_BLOCK : (X * Y);
    // How many rows can we process in parallel
    static constexpr uint32_t ROW_PER_ITERATION = RT_CONTRIBUTING_THREAD_COUNT / X;
    // How many iterations the entire image would take
    static constexpr uint32_t ROW_ITER_COUNT = Y / ROW_PER_ITERATION;
    static_assert(RT_CONTRIBUTING_THREAD_COUNT % X == 0, "RT_THREADS must be multiple of X or vice versa.");
    static_assert(Y % ROW_PER_ITERATION == 0, "RT_THREADS must exactly iterate over X * Y");
    // Ray tracing related
    static constexpr uint32_t RT_ITER_COUNT = ROW_ITER_COUNT;

    // PWC Distribution over the shared memory
    using SharedMemType = KCGenSampleShMem<THREAD_PER_BLOCK, X, Y>;
    //using Distribution2D = typename SharedMemType::BlockDist2D;
    using Filter2D = typename SharedMemType::BlockFilter2D;
    using ConeTracer = typename SharedMemType::ConeTracer;
    static_assert(RT_ITER_COUNT == ConeTracer::DATA_PER_THREAD);

    using ProductSampler8x8 = typename SharedMemType::ProductSampler8x8;

    // Functors for batched cone trace
    auto ProjectionFunc = [&](const Vector2i& localPixelId,
                              const Vector2i& segmentSize)
    {
        // Jitter the values
        Vector2f xi = rng.Uniform2D();
        Vector2f st = Vector2f(localPixelId) + xi;
        st /= Vector2f(segmentSize);
        Vector3 result = Utility::CocentricOctohedralToDirection(st);
        Vector3 dirYUp = Vector3(result[1], result[2], result[0]);
        return dirYUp;
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

    Filter2D filter(sharedMem->sFilterMem, rFieldGaussFilter, WrapFunc);
    ConeTracer coneTracer(sharedMem->sTraceMem, svo);
    // For each block (we allocate enough blocks for the GPU)
    // Each block will process multiple bins
    for(uint32_t binIndex = blockIdx.x; binIndex < binCount;
        binIndex += gridDim.x)
    {
        // Load Bin information
        if(isMainThread)
        {
            Vector2ui rayRange = Vector2ui(gBinOffsets[binIndex], gBinOffsets[binIndex + 1]);
            sharedMem->sRayCount = rayRange[1] - rayRange[0];
            sharedMem->sOffsetStart = rayRange[0];
            sharedMem->sNodeId = gNodeIds[binIndex];

            if(sharedMem->sNodeId != INVALID_BIN_ID)
            {
                // Calculate the voxel size of the bin
                uint32_t nodeId;
                bool isLeaf = ReadSVONodeId(nodeId, sharedMem->sNodeId);
                sharedMem->sBinVoxelSize = svo.NodeVoxelSize(nodeId, isLeaf);

                uint32_t randomRayIndex = static_cast<uint32_t>(rng.Uniform() * sharedMem->sRayCount);

                // TODO: Change this
                // Use the first rays hit position
                uint32_t rayId = gRayIds[rayRange[0] + randomRayIndex];
                Vector3 position = metaSurfGenerator.AcquireWork(rayId).WorldPosition();
                sharedMem->sRadianceFieldOrigin = position;
            };

            //// Roll a dice stochastically cull bin (TEST)
            //if(rng.Uniform() < 0.5f)
            //    sNodeId = INVALID_BIN_ID;
        }
        __syncthreads();

        //// Kill the entire block if Node Id is invalid
        if(sharedMem->sNodeId == INVALID_BIN_ID) continue;

        float tMin = (sharedMem->sBinVoxelSize * MathConstants::Sqrt3 +
                      MathConstants::LargeEpsilon);
        float incRadiances[RT_ITER_COUNT];

        // Batched Cone Trace
        Vector3f    rayDirOut[RT_ITER_COUNT];
        float       tMinOut[RT_ITER_COUNT];
        bool        isLeaf[RT_ITER_COUNT];
        uint32_t    nodeId[RT_ITER_COUNT];
        coneTracer.BatchedConeTraceRay(rayDirOut,
                                       tMinOut,
                                       isLeaf,
                                       nodeId,
                                       // Inputs
                                       sharedMem->sRadianceFieldOrigin,
                                       Vector2f(tMin, FLT_MAX),
                                       coneAperture, 0,
                                       // Functors
                                       ProjectionFunc);

        // Incoming radiance query (result of the trace)
        for(uint32_t i = 0; i < RT_ITER_COUNT; i++)
        {
            // This case occurs only when there is more threads than pixels
            if(THREAD_ID >= RT_CONTRIBUTING_THREAD_COUNT) continue;

            if(nodeId[i] != UINT32_MAX)
            {
                incRadiances[i] = svo.ReadRadiance(rayDirOut[i], coneAperture,
                                                   nodeId[i], isLeaf[i]);
                //printf("HitT: %f, tMin %f n:%u r:%f\n", tMinOut[i], tMin, nodeId[i], incRadiances[i]);
            }
            else incRadiances[i] = 0.0001f;
            //else incRadiances[i] = 0.0f;
        }

        //for(uint32_t i = 0; i < RT_ITER_COUNT; i++)
        //{
        //    if(THREAD_ID >= RT_CONTRIBUTING_THREAD_COUNT) continue;

        //    Vector3f hitPos = (sharedMem->sRadianceFieldOrigin +
        //                       rayDirOut[i].Normalize() * tMinOut[i]);
        //    incRadiances[i] = ReadInterpolatedRadiance(hitPos, rayDirOut[i],
        //                                               coneAperture,
        //                                               svo);
        //    if(incRadiances[i] == 0.0f) incRadiances[i] = 0.0001f;
        //}




        // We finished tracing rays from the scene
        // Now generate distribution from the data
        // and sample for each ray
        // Before that filter the radiance field
        float filteredRadiances[RT_ITER_COUNT];
        filter(filteredRadiances, incRadiances);
        __syncthreads();

        // Kernel Parallelization changes now, before that though
        ProductSampler8x8 productSampler(sharedMem->sProductSamplerMem,
                                         gRayIds, metaSurfGenerator);


        float pdf;
        Vector2f uv = productSampler.Sample(pdf, rng);

        // Generate PWC Distribution over the radiances
        //Distribution2D dist2D(sharedMem->sDistMem, filteredRadiances);
        // Block threads will loop over the every ray in this bin
        for(uint32_t rayIndex = THREAD_ID; rayIndex < sharedMem->sRayCount;
            rayIndex += THREAD_PER_BLOCK)
        {
            //float pdf;
            //Vector2f index;
            //Vector2f uv = dist2D.Sample(pdf, index, rng);
            //pdf *= 0.25f * MathConstants::InvPi;

            /*Vector3f dirZUp = Utility::CocentricOctohedralToDirection(uv);
            Vector3f dirYUp = Vector3f(dirZUp[1], dirZUp[2], dirZUp[0]);*/

            //// TEST
            //if(uv.HasNaN() || isnan(pdf))
            //    printf("[%u] uv(%f, %f) = (%f, %f, %f) pdf %f\n",
            //           binIndex, uv[0], uv[1],
            //           dirYUp[0], dirYUp[1], dirYUp[2],
            //           pdf);

            //// Store the sampled direction of the ray
            //uint32_t rayId = gRayIds[sharedMem->sOffsetStart + rayIndex];
            //gRayAux[rayId].guideDir = Vector2h(uv[0], uv[1]);
            //gRayAux[rayId].guidePDF = pdf;
        }

        // Sync every thread before processing another bin
        __syncthreads();
    }
}