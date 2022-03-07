#pragma once

#include "RayAuxStruct.cuh"

#include "GPULightI.h"
#include "GPUMediumI.h"
#include "GPUDirectLightSamplerI.h"
#include "RayStructs.h"
#include "ImageStructs.h"
#include "WorkOutputWriter.cuh"

#include "TracerFunctions.cuh"
#include "TracerConstants.h"

#include "RayLib/RandomColor.h"

#include "SceneSurfaceTreeKC.cuh"
#include "SceneSurfaceTree.cuh"
#include "QFunction.cuh"

struct RLTracerGlobalState
{
    using SpatialTree = typename SceneSurfaceTree::TreeGPUType;

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
    SpatialTree                     posTree;
    QFunctionGPU                    qFunction;
    // Options
    // Path Guiding
    bool                            rawPathGuiding;
    // Options for NEE
    bool                            directLightMIS;
    bool                            nee;
    // Russian Roulette
    int                             rrStart;
};

struct RLTracerLocalState
{
    bool emptyPrimitive;
};

template <class MGroup>
class MatFunctor
{
    using Surface = typename MGroup::Surface;
    using MatData = typename MGroup::Data;
    private:
        const MatData&      gMatData;
        const Surface&      surface;
        const Vector3f&     wi;
        const Vector3f&     position;
        uint32_t            matIndex;
        const GPUMediumI&   m;


    public:
        __device__ inline
        MatFunctor(const MatData& gMatData,
                   const Surface& surface,
                   const Vector3f& wi,
                   const Vector3f& position,
                   uint32_t matIndex,
                   const GPUMediumI& m)
            : gMatData(gMatData), surface(surface)
            , wi(wi), position(position)
            , matIndex(matIndex), m(m)
        {}

        __device__ inline
        Vector3f operator()(const Vector3f& wo) const
        {
            return MGroup::Evaluate(// Input
                                    wo,
                                    wi,
                                    position,
                                    m,
                                    //
                                    surface,
                                    // Constants
                                    gMatData,
                                    matIndex);
        }
};

template <class MGroup, class MFunctor>
__device__ inline
float AccumulateQFunction(const QFunctionGPU& qFunction,
                          uint32_t spatialIndex,
                          const MFunctor& MatEvaluate)
{
    // Accumulate all strata
    float sum = 0.0f;
    Vector2ui size = qFunction.DirectionalRes();

    for(uint32_t y = 0; y < size[1]; y++)
    for(uint32_t x = 0; x < size[0]; x++)
    {
        Vector3f wo = qFunction.Direction(Vector2ui(x, y));
        float value = qFunction.Value(Vector2ui(x, y), spatialIndex);
        Vector3f reflectance = MatEvaluate(wo);
        // TODO: Is this ok? (converting reflectance to luminance)
        // Instead of other way around
        float lumReflectance = Utility::RGBToLuminance(reflectance);
        sum += lumReflectance * value;
    }
    sum /= size.Multiply();
    sum *= 2.0f * MathConstants::Pi;
    return sum;
}

template <class EGroup>
__device__ inline
void RLTracerBoundaryWork(// Output
                           HitKey* gOutBoundKeys,
                           RayGMem* gOutRays,
                           RayAuxRL* gOutRayAux,
                           const uint32_t maxOutRay,
                           // Input as registers
                           const RayReg& ray,
                           const RayAuxRL& aux,
                           const typename EGroup::Surface& surface,
                           const RayId rayId,
                           // I-O
                           RLTracerLocalState& localState,
                           RLTracerGlobalState& renderState,
                           RNGeneratorGPUI& rng,
                           // Constants
                           const typename EGroup::GPUType& gLight)
{
    using GPUType = typename EGroup::GPUType;

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

    // And accumulate pixel and add as a sample
    Vector3f total = emission * radianceFactor;
    // Incorporate MIS weight if applicable
    // if path ray hits a light misWeight is calculated
    // else misWeight is 1.0f
    total *= misWeight;

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

        // Also  write this to the previous QFunction
        // Only do this when there is a "previous" location
        if(!isCameraRay)
        {
            float reflectance = aux.prevLumReflectance;
            float sum = reflectance * Utility::RGBToLuminance(emission);
            renderState.qFunction.Update(r.getDirection(), sum, aux.prevSpatialIndex);
        }
    }
}

template <class MGroup>
__device__ inline
void RLTracerPathWork(// Output
                      HitKey* gOutBoundKeys,
                      RayGMem* gOutRays,
                      RayAuxRL* gOutRayAux,
                      const uint32_t maxOutRay,
                      // Input as registers
                      const RayReg& ray,
                      const RayAuxRL& aux,
                      const typename MGroup::Surface& surface,
                      const RayId rayId,
                      // I-O
                      RLTracerLocalState& localState,
                      RLTracerGlobalState& renderState,
                      RNGeneratorGPUI& rng,
                      // Constants
                      const typename MGroup::Data& gMatData,
                      const HitKey::Type matIndex)
{
    using SpatialTree = RLTracerGlobalState::SpatialTree;
    const SpatialTree& posTree = renderState.posTree;

    static constexpr Vector3 ZERO_3 = Zero3;

    // TODO: change this currently only first strategy is sampled
    static constexpr int PATH_RAY_INDEX = 0;
    static constexpr int NEE_RAY_INDEX  = 1;
    // Helper Class for better code readability
    OutputWriter<RayAuxRL> outputWriter(gOutBoundKeys,
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

    // Before BxDF Acquire the 2D irradiance map
    float distance;
    SurfaceLeaf queryLeaf{position, surface.WorldNormal()};
    //uint32_t spatialIndex = posTree.FindNearestPoint(distance, queryLeaf);
    uint32_t spatialIndex = posTree.FindNearestPoint(distance, queryLeaf.position);

    if(spatialIndex == UINT32_MAX)
    {
        printf("Out of Range KdTree!\n");
        spatialIndex = 0;
    }

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
    Vector3 emission = Zero3f;
    if(MGroup::IsEmissive(gMatData, matIndex))
    {
        emission = MGroup::Emit(// Input
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
            RayAuxRL auxOut = aux;
            auxOut.radianceFactor = neeRadianceFactor;
            auxOut.endpointIndex = lightIndex;
            auxOut.type = RayType::NEE_RAY;
            auxOut.prevPDF = NAN;
            auxOut.depth++;
            // Save the RL Related Data
            auxOut.prevSpatialIndex = spatialIndex;
            auxOut.prevLumReflectance = Utility::RGBToLuminance(neeReflectance);
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
        constexpr float BxDF_PGSampleRatio = 0.0f;
        // Sample a chance
        float xi = rng.Uniform();

        bool selectedPDFZero = false;
        float pdfBxDF, pdfGuide;
        if(xi >= BxDF_PGSampleRatio)
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

            pdfGuide = 0.0f;
            pdfGuide = renderState.qFunction.Pdf(rayPath.getDirection(),
                                                 spatialIndex);
            if(pdfBxDF == 0.0f) selectedPDFZero = true;
        }
        else
        {
            // Sample a path using Path Guiding
            Vector3f direction = renderState.qFunction.Sample(pdfGuide, rng,
                                                              spatialIndex);
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

            if(pdfGuide == 0.0f) selectedPDFZero = true;
        }
        // Pdf Average
        pdfPath = BxDF_PGSampleRatio          * pdfGuide +
                  (1.0f - BxDF_PGSampleRatio) * pdfBxDF;
        pdfPath = selectedPDFZero ? 0.0f : pdfPath;

        // DEBUG
        if(isnan(pdfPath) || isnan(pdfBxDF) || isnan(pdfGuide))
            printf("[%s] NAN PDF = % f = w * %f + (1.0f - w) * %f, w: % f\n",
                   (xi >= BxDF_PGSampleRatio) ? "BxDF": "Tree",
                   pdfPath, pdfGuide, pdfBxDF, BxDF_PGSampleRatio);
        if(pdfPath != 0.0f && rayPath.getDirection().HasNaN())
            printf("[%s] NAN DIR %f, %f, %f\n",
                    (xi >= BxDF_PGSampleRatio) ? "BxDF" : "Tree",
                    rayPath.getDirection()[0],
                    rayPath.getDirection()[1],
                    rayPath.getDirection()[2]);
        if(reflectance.HasNaN())
            printf("[%s] NAN REFL %f %f %f\n",
                   (xi >= BxDF_PGSampleRatio) ? "BxDF" : "Tree",
                   reflectance[0],
                   reflectance[1],
                   reflectance[2]);

        //if(isnan(pdfPath) || isnan(pdfBxDF) || isnan(pdfTree) ||
        //   rayPath.getDirection().HasNaN() || reflectance.HasNaN())
        //    return;

        // If this ray is not camera ray
        // Accumulate everything on the current spatial data
        // and send it to previous spatial data
        if(aux.type != RayType::CAMERA_RAY)
        {
            MatFunctor<MGroup> MatEval(gMatData, surface,
                                       wi, position,
                                       matIndex, m);

            float sum = AccumulateQFunction<MGroup>(renderState.qFunction,
                                                    spatialIndex,
                                                    MatEval);
            // Don't forget to add current emission if available
            sum += Utility::RGBToLuminance(emission);
            // Atomically update the value in the previous Q function
            renderState.qFunction.Update(wi, sum, aux.prevSpatialIndex);
        }
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

        // Since material is a specular material
        // Only send the found direction value
        // as if the accumulated value
        float value = renderState.qFunction.Value(rayPath.getDirection(), spatialIndex);
        float sum = Utility::RGBToLuminance(reflectance) * value;
        renderState.qFunction.Update(wi, sum, aux.prevSpatialIndex);
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
        RayAuxRL auxOut = aux;
        auxOut.mediumIndex = static_cast<uint16_t>(outM->GlobalIndex());
        auxOut.radianceFactor = pathRadianceFactor;
        auxOut.type = (isSpecularMat) ? RayType::SPECULAR_PATH_RAY : RayType::PATH_RAY;
        // Save BxDF pdf if we hit a light
        // When we hit a light we will
        auxOut.prevPDF = pdfPath;
        auxOut.depth++;
        // Save the RL Related Data
        auxOut.prevSpatialIndex = spatialIndex;
        auxOut.prevLumReflectance = Utility::RGBToLuminance(reflectance);

        // Write
        outputWriter.Write(PATH_RAY_INDEX, rayOut, auxOut);
    }
    // All Done!
}

template <class EGroup>
__device__ inline
void RLTracerDebugBWork(// Output
                        HitKey* gOutBoundKeys,
                        RayGMem* gOutRays,
                        RayAuxRL* gOutRayAux,
                        const uint32_t maxOutRay,
                        // Input as registers
                        const RayReg& ray,
                        const RayAuxRL& aux,
                        const typename EGroup::Surface& surface,
                        const RayId rayId,
                        // I-O
                        RLTracerLocalState& localState,
                        RLTracerGlobalState& renderState,
                        RNGeneratorGPUI& rng,
                        // Constants
                        const typename EGroup::GPUType& gLight)
{
    using SpatialTree = RLTracerGlobalState::SpatialTree;
    const SpatialTree& posTree = renderState.posTree;

    // Helper Class for better code readability
    OutputWriter<RayAuxRL> outputWriter(gOutBoundKeys,
                                        gOutRays,
                                        gOutRayAux,
                                        maxOutRay);

    // Only Direct Hits from camera are used
    // In debugging
    if(aux.depth != 1) return;

    // Inputs
    // Current Ray
    const RayF& r = ray.ray;
    // Hit Position
    Vector3 position = r.AdvancedPos(ray.tMax);

    // Acquire Spatial Loc
    float distance;
    SurfaceLeaf queryLeaf{position, surface.WorldNormal()};
    //uint32_t spatialIndex = posTree.FindNearestPoint(distance, queryLeaf);
    uint32_t spatialIndex = posTree.FindNearestPoint(distance, queryLeaf.position);
    Vector3f locColor = (distance <= posTree.VoronoiCenterSize())
                         ? Zero3f
                         : Utility::RandomColorRGB(spatialIndex);

    // Accumulate the pixel
    ImageAccumulatePixel(renderState.gImage,
                         aux.pixelIndex,
                         Vector4f(locColor, 1.0f));
}

template <class MGroup>
__device__ inline
void RLTracerDebugWork(// Output
                       HitKey* gOutBoundKeys,
                       RayGMem* gOutRays,
                       RayAuxRL* gOutRayAux,
                       const uint32_t maxOutRay,
                       // Input as registers
                       const RayReg& ray,
                       const RayAuxRL& aux,
                       const typename MGroup::Surface& surface,
                       const RayId rayId,
                       // I-O
                       RLTracerLocalState& localState,
                       RLTracerGlobalState& renderState,
                       RNGeneratorGPUI& rng,
                       // Constants
                       const typename MGroup::Data& gMatData,
                       const HitKey::Type matIndex)
{
    using SpatialTree = RLTracerGlobalState::SpatialTree;
    const SpatialTree& posTree = renderState.posTree;
    // Helper Class for better code readability
    OutputWriter<RayAuxRL> outputWriter(gOutBoundKeys,
                                        gOutRays,
                                        gOutRayAux,
                                        maxOutRay);

    // Only Direct Hits from camera are used
    // In debugging
    if(aux.depth != 1) return;

    // Inputs
    // Current Ray
    const RayF& r = ray.ray;
    // Hit Position
    Vector3 position = r.AdvancedPos(ray.tMax);

    // Acquire Spatial Loc
    float distance;
    SurfaceLeaf queryLeaf{position, surface.WorldNormal()};
    //uint32_t spatialIndex = posTree.FindNearestPoint(distance, queryLeaf);
    uint32_t spatialIndex = posTree.FindNearestPoint(distance, queryLeaf.position);
    //Vector3f locColor = (distance <= posTree.VoronoiCenterSize())
    //                     ? Zero3f
    //                     : Utility::RandomColorRGB(spatialIndex);
    Vector3f locColor = Utility::RandomColorRGB(spatialIndex);
    // Accumulate the pixel
    ImageAccumulatePixel(renderState.gImage,
                         aux.pixelIndex,
                         Vector4f(locColor, 1.0f));
}