#include  "GPUAcceleratorOptixKC.cuh"

#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

struct RayPayload
{
    HitKey*         gRayWorkKey;
    TransformId*    gRayTransformId;
    PrimitiveId*    gRayPrimitiveId;
    HitStructPtr    gRayHitStruct;
};

extern "C" __constant__ OpitXBaseAccelParams params;

extern "C" __global__ void KCRayGenOptix()
{
    // We Launch linearly
    const uint32_t theLaunchDim = optixGetLaunchDimensions().x;
    const uint32_t theLaunchIndex = optixGetLaunchIndex().x;

    // Load Ray
    RayReg ray(params.gRays, theLaunchIndex);

    // We can Store everything inside the ray payload
    // but is it performant? I dunno
    // I will do the tutorial style (Optix 7 course etc.)
    // local memory struct pointer as ray payload
    RayPayload pointers;

    pointers.gRayWorkKey = params.gWorkKeys + theLaunchIndex;
    pointers.gRayTransformId = params.gTransformIds + theLaunchIndex;
    pointers.gRayPrimitiveId = params.gPrimitiveIds + theLaunchIndex;
    pointers.gRayHitStruct = params.gHitStructs.AdvancedPtr(theLaunchIndex);
    // Pointer to Payload
    uint2 payloadPtr = PointerToUInt2<RayPayload>(&pointers);

    optixTrace(// Accelrator
               params.baseAcceleratorOptix,
               // Ray Input
               make_float3(ray.ray.getPosition()[0],
                           ray.ray.getPosition()[1],
                           ray.ray.getPosition()[2]),
               make_float3(ray.ray.getDirection()[0],
                           ray.ray.getDirection()[1],
                           ray.ray.getDirection()[2]),
               ray.tMin,
               ray.tMax,
               0.0f,
               //
               OptixVisibilityMask(255),
               // Flags
               OPTIX_RAY_FLAG_NONE,
               // SBT
               0, 1, 0,
               // Payload
               payloadPtr.x, payloadPtr.y);

    // All should be written
}

// Meta Closest Hit Shader
template<class PGroup>
__device__ void KCClosestHit()
{
    using HitStruct = typename PGroup::HitData;
    using Record = typename HitGroupRecord<typename PGroup::PrimitiveData,
                                           typename PGroup::LeafData>::Data;
    const Record& r = *(const Record*)optixGetSbtDataPointer();

    const int leafId = optixGetPrimitiveIndex();

    RayPayload* lPayload = UInt2ToPointer<RayPayload>({optixGetPayload_0(),
                                                    optixGetPayload_1()});

    (*lPayload->gRayPrimitiveId) = r.gLeafs[leafId].primitiveId;
    (*lPayload->gRayTransformId) = r.transformId;
    (*lPayload->gRayWorkKey) = r.gLeafs[leafId].matId;

    // Set Hit Struct
    // PGroup::Alp

    float2 baryCoords = optixGetTriangleBarycentrics();
    lPayload->gRayHitStruct.Ref<HitStruct>(0) = Vector2f(baryCoords.x,
                                                         baryCoords.y);
}

// Meta Any Hit Shader
template<class PGroup>
__device__ void KCAnyHit()
{
    using HitStruct = typename PGroup::HitData;
    using Record = typename HitGroupRecord<typename PGroup::PrimitiveData,
                                           typename PGroup::LeafData>::Data;
    const Record& r = *(const Record*)optixGetSbtDataPointer();

    const int leafId = optixGetPrimitiveIndex();

    // Call Alpha Check
    float newT;

    //bool missed = PGroup::AlphaTest();

    //if(missed) optixIgnoreIntersection();
}

template<class PGroup>
__device__ void KCIntersect()
{
    using HitStruct = typename PGroup::HitData;
    using Record = typename HitGroupRecord<typename PGroup::PrimitiveData,
                                           typename PGroup::LeafData>::Data;
    const Record& r = *(const Record*)optixGetSbtDataPointer();

    const int leafId = optixGetPrimitiveIndex();

    // Do Intersect

    //
    //optixReportIntersection(
    //    t * l,
    //    HIT_OUTSIDE_FROM_INSIDE,
    //    float3_as_args(normal));
}

// Actual Definitions
// Using this style instead of demangling function names etc.
// Triangle
WRAP_FUCTION(GPUPrimitiveTriangle_KCClosestHit, KCClosestHit<GPUPrimitiveTriangle>)
WRAP_FUCTION(GPUPrimitiveTriangle_KCAnyHit, KCClosestHit<GPUPrimitiveTriangle>)

// Sphere
WRAP_FUCTION(GPUPrimitiveSphere_KCClosestHit, KCClosestHit<GPUPrimitiveSphere>)
WRAP_FUCTION(GPUPrimitiveSphere_KCAnyHit, KCClosestHit<GPUPrimitiveSphere>)
WRAP_FUCTION(GPUPrimitiveSphere_KCIntersect, KCClosestHit<GPUPrimitiveSphere>)

//// Define a MACRO
//extern "C" __global__ void TEST_CLOSEST_HIT()
//{
//    KCClosestHit<GPUPrimitiveTriangle>();
//}
