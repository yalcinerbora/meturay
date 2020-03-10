#pragma once

#include "RayAuxStruct.h"

#include "TracerLib/MaterialFunctions.cuh"
#include "TracerLib/ImageStructs.h"
#include "TracerLib/RayStructs.h"
#include "TracerLib/TextureStructs.h"

struct BasicTracerGlobal
{
    ImageGMem<Vector4> gImage;
};

// No Local State
struct BasicTracerLocal {};

template <class MGroup>
__device__
inline void BasicWork(// Output
                      HitKey* gOutBoundKeys,
                      RayGMem* gOutRays,
                      RayAuxBasic* gOutRayAux,
                      const uint32_t maxOutRay,
                      // Input as registers                         
                      const RayReg& ray,
                      const RayAuxBasic& aux,
                      const MGroup::Surface& surface,
                      const UVList* uvs,
                      // I-O
                      BasicTracerLocal& gLocalState,
                      BasicTracerGlobal& gRenderState,
                      RandomGPU& rng,
                      // Constants
                      const MGroup::Data& gMatData,
                      const HitKey::Type& matId)
{
    //MGroup::Sample()
}