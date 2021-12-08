#pragma once

#include "RayLib/HitStructs.h"
#include "RayStructs.h"
#include "GPUTransformI.h"

#include <optix_device.h>

struct OpitXBaseAccelParams
{
    OptixTraversableHandle baseAcceleratorOptix;

    // Outputs
    HitKey*                 gWorkKeys;
    TransformId*            gTransformIds;
    PrimitiveId*            gPrimitiveIds;
    HitStructPtr            gHitStructs;
    // I-O
    RayGMem*                gRays;
    // Constants
    uint32_t                maxAttributeCount;
    const GPUTransformI**   gGlobalTransformArray;
};

template <class PrimData, class LeafStruct>
struct Record
{
    // This pointer is can be accessed by optixGetPrimitiveIndex() from Optix
    const LeafStruct* gLeafs;
    // Each Accelerator has its own accelerator Id
    TransformId transformId;
    // PrimData holds the global info of the primitive
    // Data inside should be accessed as such:
    //   int leafId = optixGetPrimitiveIndex();
    //   (*primData).positions[gLeafs[leafId].primitiveId]
    const PrimData* gPrimData;
};

// Meta Hit Record
// This Record is kept for each accelerator in an accelerator group
template <class PrimData, class LeafStruct>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

    Record<PrimData, LeafStruct> data;
};

// Other SBT Records are Empty (use this for those API calls)
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Optix 7 Course had dummy pointer here so i will leave it as well
    void* data;
};

// ExternCWrapper Macro
#define WRAP_FUCTION(NAME, FUNCTION) \
    extern "C" __global__ void NAME(){FUNCTION();}
