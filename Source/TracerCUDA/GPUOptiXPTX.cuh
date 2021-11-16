#pragma once

#include "RayLib/HitStructs.h"
#include "RayStructs.h"

#include <optix_device.h>

struct OpitXBaseAccelParams
{
    OptixTraversableHandle baseAcceleratorOptix;

    // Outputs
    HitKey*         gWorkKeys;
    TransformId*    gTransformIds;
    PrimitiveId*    gPrimitiveIds;
    HitStructPtr    gHitStructs;
    // Inputs
    const RayGMem*  gRays;
};

// Meta Hit Record
// This Record is kept for each accelerator in an accelerator group
template <class PrimData, class LeafStruct>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

    struct Data
    {
        // This pointer is can be accessed by optixGetPrimitiveIndex() from Optix
        LeafStruct* gLeafs;
        // Each Accelerator has its own accelerator Id
        TransformId transformId;
        // PrimData holds the global info of the primitive
        // Data inside should be accesed as such:
        //   int leafId = optixGetPrimitiveIndex();
        //   (*primData).positions[gLeafs[leafId].primitiveId]
        PrimData* primData;
    };
};

// Other SBT Records are Empty (use this for those API calls)
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Optix 7 Course had dummy pointer here so i will leave it aswell
    void* data;
};

// Meta Functions
template<class T>
__forceinline__ __device__
T* UInt2ToPointer(const uint2& ptrUInt2)
{
    static_assert(sizeof(T*) == sizeof(uint64_t));
    uint64_t ptr = static_cast<uint64_t>(ptrUInt2.x) << 32;
    ptr |= static_cast<uint64_t>(ptrUInt2.y);
    return reinterpret_cast<T*>(ptr);
}

template<class T>
__forceinline__ __device__
uint2 PointerToUInt2(T* ptr)
{
    static_assert(sizeof(T*) == sizeof(uint64_t), "");
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    return make_uint2(uptr >> 32,
                      uptr & 0x00000000ffffffff);
}

// ExternCWrapper Macro
#define WRAP_FUCTION(NAME, FUNCTION) \
    extern "C" __global__ void NAME(){FUNCTION();}