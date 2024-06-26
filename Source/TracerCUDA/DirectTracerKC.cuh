#pragma once

#include "RayStructs.h"
#include "RayAuxStruct.cuh"
#include "ImageStructs.h"
#include "GPUMediumVacuum.cuh"
#include "GPUCameraI.h"
#include "DebugMaterials.cuh"
#include "EmptyMaterial.cuh"

#include "RayLib/HitStructs.h"

enum class NormalRenderType
{
    WORLD,
    LOCAL,
    SURFACE
};

enum class PositionRenderType
{
    VECTOR3,
    LINEAR_DEPTH,
    LOG_DEPTH
};

struct DirectTracerGlobalState
{
    // Samples
    CamSampleGMem<Vector4f> gSamples;
};

// Position Render Specific States
struct DirectTracerPositionGlobalState : public DirectTracerGlobalState
{
    PositionRenderType  posRenderType;
    const GPUCameraI*   gCurrentCam;
};

// No Local State
struct DirectTracerLocalState {};

template <class EGroup>
__device__ inline
void DirectBoundaryWork(// Output
                        HitKey* gOutBoundKeys,
                        RayGMem* gOutRays,
                        RayAuxBasic* gOutRayAux,
                        const uint32_t maxOutRay,
                        // Input as registers
                        const RayReg& ray,
                        const RayAuxBasic& aux,
                        const typename EGroup::Surface& surface,
                        const RayId rayId,
                        // I-O
                        DirectTracerLocalState& localState,
                        DirectTracerGlobalState& renderState,
                        RNGeneratorGPUI& rng,
                        // Constants
                        const typename EGroup::GPUType& gLight)
{
    const RayF& r = ray.ray;
    Vector3 position = surface.WorldPosition();

    Vector3 emission = gLight.Emit(// Input
                                   -r.getDirection(),
                                   position,
                                   surface);
    AccumulateRaySample(renderState.gSamples,
                        aux.sampleIndex,
                        Vector4f(emission, 1.0f));
}

template <class MGroup>
__device__
inline void DirectFurnaceWork(// Output
                              HitKey* gOutBoundKeys,
                              RayGMem* gOutRays,
                              RayAuxBasic* gOutRayAux,
                              const uint32_t maxOutRay,
                              // Input as registers
                              const RayReg& ray,
                              const RayAuxBasic& aux,
                              const typename MGroup::Surface& surface,
                              const RayId rayId,
                              // I-O
                              DirectTracerLocalState& localState,
                              DirectTracerGlobalState& renderState,
                              RNGeneratorGPUI& rng,
                              // Constants
                              const typename MGroup::Data& gMatData,
                              const HitKey::Type matIndex)
{
    // Just evaluate kernel
    // Write to image
    const RayF& r = ray.ray;

    const GPUMediumVacuum m(0);
    const GPUMediumI* outM;
    RayF outRay; float pdf;

    Vector3 emission = MGroup::Emit(-r.getDirection(),
                                    surface.WorldPosition(),
                                    m,
                                    //
                                    surface,
                                    //
                                    gMatData,
                                    matIndex);

    Vector3 radiance = MGroup::Sample(// Outputs
                                      outRay, pdf, outM,
                                      // Inputs
                                      -r.getDirection(),
                                      surface.WorldPosition(),
                                      m,
                                      //
                                      surface,
                                      // I-O
                                      rng,
                                      // Constants
                                      gMatData,
                                      matIndex,
                                      0);

    radiance = (pdf == 0.0f) ? Zero3 : (radiance / pdf);
    radiance += emission;
    // And accumulate pixel
    AccumulateRaySample(renderState.gSamples,
                        aux.sampleIndex,
                        Vector4(radiance, 1.0f));
}

__device__
inline void DirectPositionWork(// Output
                               HitKey* gOutBoundKeys,
                               RayGMem* gOutRays,
                               RayAuxBasic* gOutRayAux,
                               const uint32_t maxOutRay,
                               // Input as registers
                               const RayReg& ray,
                               const RayAuxBasic& aux,
                               const typename EmptyMat<EmptySurface>::Surface& surface,
                               const RayId rayId,
                               // I-O
                               DirectTracerLocalState& localState,
                               DirectTracerPositionGlobalState& renderState,
                               RNGeneratorGPUI& rng,
                               // Constants
                               const typename EmptyMat<EmptySurface>::Data& gMatData,
                               const HitKey::Type matIndex)
{
    static constexpr float C = 1.0f;

    Vector4f worldPos = Vector4f(ray.ray.getPosition() + ray.ray.getDirection() * ray.tMax,
                                 1.0f);
    switch(renderState.posRenderType)
    {
        case PositionRenderType::VECTOR3:
        {
            AccumulateRaySample(renderState.gSamples,
                                aux.sampleIndex,
                                worldPos);
            return;
        }
        case PositionRenderType::LINEAR_DEPTH:
        case PositionRenderType::LOG_DEPTH:
        {
            float depth;
            Vector2f nearFar = renderState.gCurrentCam->NearFar();
            Vector4f ndc = renderState.gCurrentCam->VPMatrix() * worldPos;

            if(renderState.posRenderType == PositionRenderType::LINEAR_DEPTH)
                depth = ndc[3];
            else
                depth = log(C * ndc[3] + 1.0f) / log(C * nearFar[1] + 1.0f);

            AccumulateRaySample(renderState.gSamples,
                                aux.sampleIndex,
                                Vector4(depth, depth, depth, 1.0f));
            break;
        }
    }

}

__device__
inline void DirectNormalWork(// Output
                             HitKey* gOutBoundKeys,
                             RayGMem* gOutRays,
                             RayAuxBasic* gOutRayAux,
                             const uint32_t maxOutRay,
                             // Input as registers
                             const RayReg& ray,
                             const RayAuxBasic& aux,
                             const typename NormalRenderMat::Surface& surface,
                             const RayId rayId,
                             // I-O
                             DirectTracerLocalState& localState,
                             DirectTracerGlobalState& renderState,
                             RNGeneratorGPUI& rng,
                             // Constants
                             const typename NormalRenderMat::Data& gMatData,
                             const HitKey::Type matIndex)
{
    Vector3f ZERO = Zero3;
    const GPUMediumVacuum m(0);
    Vector3f normal = NormalRenderMat::Evaluate(ZERO, ZERO, ZERO, m, surface, gMatData, matIndex);
    AccumulateRaySample(renderState.gSamples,
                        aux.sampleIndex,
                        Vector4(normal, 1.0f));
}