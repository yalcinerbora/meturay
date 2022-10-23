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
            Vector2f thetaPhi = Vector2f(// [-pi, pi]
                                         (uv[0] * MathConstants::Pi * 2.0f) - MathConstants::Pi,
                                          // [0, pi]
                                         (1.0f - uv[1]) * MathConstants::Pi);
            Vector3f dirZUp = Utility::SphericalToCartesianUnit(thetaPhi);
            Vector3f direction = Vector3f(dirZUp[1], dirZUp[2], dirZUp[0]);
            // Convert to solid angle pdf
            // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources.html
            pdfGuide = aux.guidePDF;
            float sinPhi = sin(thetaPhi[1]);
            if(sinPhi == 0.0f) pdfGuide = 0.0f;
            else pdfGuide = pdfGuide / (2.0f * MathConstants::Pi * MathConstants::Pi * sinPhi);

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
    bool found = svo.LeafIndex(svoLeafIndex, position, true);
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
    bool found = svo.LeafIndex(svoLeafIndex, position, true);
    Vector3f locColor = (found) ? Utility::RandomColorRGB(svoLeafIndex)
                                : Vector3f(0.0f);
    // Accumulate the pixel
    AccumulateRaySample(renderState.gSamples,
                        aux.sampleIndex,
                        Vector4f(locColor, 1.0f));
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCTraceSVO(// Output
                       CamSampleGMem<Vector4f> gSamples,
                       // Input
                       const AnisoSVOctreeGPU svo,
                       const RayGMem* gRays,
                       const RayAuxWFPG* gRayAux,
                       // Constants
                       const float coneAperture,
                       WFPGRenderMode mode,
                       uint32_t maxQueryLevelOffset,
                       uint32_t rayCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < rayCount;
        threadId += (blockDim.x * gridDim.x))
    {
        RayReg ray = RayReg(gRays, threadId);
        RayAuxWFPG aux = gRayAux[threadId];

        bool isLeaf;
        uint32_t svoNodeIndex;
        float tMin = svo.ConeTraceRay(isLeaf, svoNodeIndex, ray.ray,
                                      ray.tMin, ray.tMax,
                                      coneAperture,
                                      maxQueryLevelOffset);

        Vector4f locColor = Vector4f(0.0f, 0.0f, 10.0f, 1.0f);
        // Octree Display Mode
        if(mode == WFPGRenderMode::SVO_FALSE_COLOR)
            locColor = (svoNodeIndex != UINT32_MAX) ? Vector4f(Utility::RandomColorRGB(svoNodeIndex), 1.0f)
                                                    : Vector4f(Vector3f(0.0f), 1.0f);
        // Payload Display Mode
        else if(svoNodeIndex == UINT32_MAX)
            locColor = Vector4f(1.0f, 0.0f, 1.0f, 1.0f);
        else if(mode == WFPGRenderMode::SVO_RADIANCE)
        {
            Vector3f coneDir = ray.ray.getDirection();
            half radiance = svo.ReadRadiance(coneDir, coneAperture,
                                             svoNodeIndex, isLeaf);
            float radianceF = radiance;
            if(radiance != static_cast<half>(MRAY_HALF_MAX))
                locColor = Vector4f(Vector3f(radianceF), 1.0f);
        }
        else if(mode == WFPGRenderMode::SVO_NORMAL)
        {
            float stdDev;
            Vector3f normal = svo.DebugReadNormal(stdDev, svoNodeIndex, isLeaf);

            // Voxels are two sided show the normal for the current direction
            normal = (normal.Dot(ray.ray.getDirection()) >= 0.0f) ? normal : -normal;

            // Convert normal to 0-1 range
            //normal += Vector3f(1.0f);
            //normal *= Vector3f(0.5f);
            locColor = Vector4f(normal, stdDev);
        }
        // Accumulate the pixel
        AccumulateRaySample(gSamples,
                            aux.sampleIndex,
                            locColor);
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
        bool found = svo.LeafIndex(leafIndex, position, true);

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

// Shared Memory Class of the kernel below
template <uint32_t THREAD_PER_BLOCK, uint32_t X, uint32_t Y>
struct KCGenSampleShMem
{
    // PWC Distribution over the shared memory
    using BlockDist2D = BlockPWCDistribution2D<THREAD_PER_BLOCK, X, Y>;
    //using BlockDist2D = BlockPWLDistribution2D<THREAD_PER_BLOCK, X, Y>;
    using BlockFilter2D = BlockTextureFilter2D<GaussFilter, THREAD_PER_BLOCK, X, Y>;
    union
    {
        typename BlockDist2D::TempStorage   sDistMem;
        typename BlockFilter2D::TempStorage sFilterMem;
    };
    // Starting positions of the rays (at most TPB)
    Vector3f sPositions[THREAD_PER_BLOCK];
    //Vector3f sPosition;
    // Bin parameters
    uint32_t sRayCount;
    uint32_t sOffsetStart;
    uint32_t sNodeId;
    uint32_t sPositionCount;
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
    using Distribution2D = typename SharedMemType::BlockDist2D;
    using Filter2D = typename SharedMemType::BlockFilter2D;

    // Change the type of the shared memory
    extern __shared__ Byte sharedMemRAW[];
    SharedMemType* sharedMem = reinterpret_cast<SharedMemType*>(sharedMemRAW);

    Filter2D filter(sharedMem->sFilterMem, rFieldGaussFilter,
                    Utility::CocentricOctohedralWrapInt);

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
            sharedMem->sPositionCount = min(THREAD_PER_BLOCK, sharedMem->sRayCount);

            // Calculate the voxel size of the bin
            uint32_t nodeId;
            bool isLeaf = ReadSVONodeId(nodeId, sharedMem->sNodeId);
            sharedMem->sBinVoxelSize = svo.NodeVoxelSize(nodeId, isLeaf);

            //// Roll a dice stochastically cull bin (TEST)
            //if(rng.Uniform() < 0.5f)
            //    sNodeId = INVALID_BIN_ID;
        }
        __syncthreads();

        // Kill the entire block if Node Id is invalid
        if(sharedMem->sNodeId == INVALID_BIN_ID) continue;

        // Load positions of the rays to the shared memory
        // Try to load minimum of TPB or ray count
        if(THREAD_ID < sharedMem->sPositionCount)
        {
            uint32_t rayId = gRayIds[sharedMem->sOffsetStart + THREAD_ID];
            RayReg ray = metaSurfGenerator.Ray(rayId);
            // Hit Position
            Vector3 position = ray.ray.AdvancedPos(ray.tMax);
            sharedMem->sPositions[THREAD_ID] = position;
        }
        __syncthreads();

        // Incoming radiance (result of the trace)
        float incRadiances[RT_ITER_COUNT];

        // Trace the rays
        for(uint32_t i = 0; i < RT_ITER_COUNT; i++)
        {
            // This case occurs only when there is more threads than pixels
            if(THREAD_ID >= RT_CONTRIBUTING_THREAD_COUNT) continue;

            // Determine your direction
            uint32_t directionId = (i * THREAD_PER_BLOCK) + THREAD_ID;
            // Make directions similar for tracing
            // Use morton code to order
            Vector2ui dirIdXY = MortonCode::Decompose2D(directionId);
            // ONLY WORKS ON POW OF 2 TEX
            //Vector2ui dirIdXY = Vector2ui(directionId % X,
            //                              directionId / X);
            Vector3f worldDir = DirIdToWorldDir(dirIdXY, Vector2ui(X, Y), rng);
            // Get a random position from the pool
            Vector3f position = sharedMem->sPositions[THREAD_ID % sharedMem->sPositionCount];
            //Vector3f position = sPositions[0];

            // Now this is the interesting part
            // We need to offset the ray in order to prevent
            // self intersections.
            // However it is not easy since we use arbitrary locations
            // over the voxel, most fool-proof (but inaccurate)
            // way is to offset the ray with the current level voxel size.
            // This will be highly inaccurate when current bin(node) level is low.
            // TODO: Change it to a better solution
            float tMin = (sharedMem->sBinVoxelSize * MathConstants::Sqrt3 +
                          MathConstants::LargeEpsilon);

            bool isLeaf;
            uint32_t nodeId;
            float hitT = svo.ConeTraceRay(isLeaf, nodeId, RayF(worldDir, position),
                                          tMin, FLT_MAX, coneAperture);

            // TODO: change this
            float radiance;
            if(nodeId != UINT32_MAX)
            {
                radiance = svo.ReadRadiance(worldDir, coneAperture,
                                            nodeId, isLeaf);
                //printf("HitT: %f, tMin %f n:%u r:%f\n", hitT, tMin, nodeId, radiance);
            }
            else radiance = 0.0001f;
            //else radiance = 2.0f;

            incRadiances[i] = radiance;
        }
        // We finished tracing rays from the scene
        // Now generate distribution from the data
        // and sample for each ray

        // Before that filter the radiance field
        float filteredRadiances[RT_ITER_COUNT];
        filter(filteredRadiances, incRadiances);
        __syncthreads();

        // Generate PWC Distribution over the radiances
        Distribution2D dist2D(sharedMem->sDistMem, filteredRadiances);
        // Block threads will loop over the every ray in this bin
        for(uint32_t rayIndex = THREAD_ID; rayIndex < sharedMem->sRayCount;
            rayIndex += THREAD_PER_BLOCK)
        {
            float pdf;
            Vector2f index;
            Vector2f uv = dist2D.Sample(pdf, index, rng);

            //Vector2f thetaPhi = Vector2f(// [-pi, pi]
            //                             (uv[0] * MathConstants::Pi * 2.0f) - MathConstants::Pi,
            //                              // [0, pi]
            //                             (1.0f - uv[1]) * MathConstants::Pi);
            //Vector3f dirZUp = Utility::SphericalToCartesianUnit(thetaPhi);
            //Vector3f debugDir = Vector3f(dirZUp[1], dirZUp[2], dirZUp[0]);
            //printf("[%u] uv(%f, %f) = (%f, %f, %f) pdf %f\n",
            //       binIndex, uv[0], uv[1],
            //       debugDir[0], debugDir[1], debugDir[2],
            //       pdf);
            // Store the sampled direction of the ray
            uint32_t rayId = gRayIds[sharedMem->sOffsetStart + rayIndex];
            gRayAux[rayId].guideDir = Vector2h(uv[0], uv[1]);
            gRayAux[rayId].guidePDF = pdf;
        }

        // Sync every thread before processing another bin
        __syncthreads();
    }
}

__device__
static float GenConeAperture(const GPUCameraI& gCamera,
                             const Vector2i& resolution)
{
    Vector2f fov = gCamera.FoV();
    float coneAngleX = fov[0] / static_cast<float>(resolution[0]);
    float coneAngleY = fov[1] / static_cast<float>(resolution[1]);
    // Return average
    return (coneAngleX + coneAngleY) * 0.5f;
}

__global__
static void KCGenConeApertureFromObject(// Output
                                        float& gConeAperture,
                                        // Input
                                        const GPUCameraI& gCamera,
                                        // Constants
                                        Vector2i resolution)
{

    if(threadIdx.x != 0) return;
    gConeAperture = GenConeAperture(gCamera, resolution);
}

__global__
static void KCGenConeApertureFromArray(// Output
                                       float& gConeAperture,
                                       // Input
                                       const GPUCameraI** gCameras,
                                       const uint32_t sceneCamId,
                                       // Constants
                                       Vector2i resolution)
{
    if(threadIdx.x != 0) return;
    gConeAperture = GenConeAperture(*gCameras[sceneCamId], resolution);
}