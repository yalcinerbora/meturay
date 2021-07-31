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
    ImageGMem<Vector4> gImage;
};

// No Local State
struct DirectTracerLocalState {};

struct DirectTracerPositionLocalState 
{
    PositionRenderType posRenderType;
    GPUCameraI* gCurrentCam;
};

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
                              DirectTracerLocalState& gLocalState,
                              DirectTracerGlobalState& gRenderState,
                              RandomGPU& rng,
                              // Constants
                              const typename MGroup::Data& gMatData,
                              const HitKey matId,
                              const PrimitiveId primId)
{
    // Just evaluate kernel
    // Write to image
    auto& img = gRenderState.gImage;
    const RayF& r = ray.ray;
    HitKey::Type matIndex = HitKey::FetchIdPortion(matId);

    const GPUMediumVacuum m(0);
    const GPUMediumI* outM;
    RayF outRay; float pdf;

    Vector3 emission = MGroup::Emit(-r.getDirection(),
                                    r.AdvancedPos(ray.tMax),
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
                                      r.AdvancedPos(ray.tMax),
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

    //printf("r(%f, %f, %f), e(%f, %f, %f), pdf %f\n",
    //       radiance[0], radiance[1], radiance[2],
    //       emission[0], emission[1], emission[2],
    //       pdf);

    // And accumulate pixel
    ImageAccumulatePixel(img, aux.pixelIndex, Vector4(radiance, 1.0f));
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
                               DirectTracerPositionLocalState& gLocalState,
                               DirectTracerGlobalState& gRenderState,
                               RandomGPU& rng,
                               // Constants
                               const typename EmptyMat<EmptySurface>::Data& gMatData,
                               const HitKey matId,
                               const PrimitiveId primId)
{   
    static constexpr float C = 1.0f;

    auto& img = gRenderState.gImage;
    const RayF& r = ray.ray;
    Vector4f worldPos = Vector4f(r.AdvancedPos(ray.tMax), 1.0f);
    switch(gLocalState.posRenderType)
    {
        case PositionRenderType::VECTOR3:
        {            
            ImageAccumulatePixel(img, aux.pixelIndex, worldPos);
            return;
        }
        case PositionRenderType::LINEAR_DEPTH:
        case PositionRenderType::LOG_DEPTH:
        {
            float depth;
            Vector2f nearFar = gLocalState.gCurrentCam->NearFar();
            Vector4f ndc = worldPos * gLocalState.gCurrentCam->VPMatrix();

            if(gLocalState.posRenderType == PositionRenderType::LINEAR_DEPTH)
                depth = ndc[3];
            else
                depth = log(C * ndc[3] + 1.0f) / log(C * nearFar[1] + 1.0f);

            ImageAccumulatePixel(img, aux.pixelIndex, Vector4(depth, 0.0f, 0.0f, 1.0f));
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
                               DirectTracerLocalState& gLocalState,
                               DirectTracerGlobalState& gRenderState,
                               RandomGPU& rng,
                               // Constants
                               const typename NormalRenderMat::Data& gMatData,
                               const HitKey matId,
                               const PrimitiveId primId)
{
    auto& img = gRenderState.gImage;
    Vector3f ZERO = Zero3;
    const GPUMediumVacuum m(0);
    HitKey::Type matIndex = HitKey::FetchIdPortion(matId);
    Vector3f normal = NormalEvaluate(ZERO, ZERO, ZERO, m, surface, gMatData, matIndex);    
    ImageAccumulatePixel(img, aux.pixelIndex, Vector4(normal, 1.0f));
}