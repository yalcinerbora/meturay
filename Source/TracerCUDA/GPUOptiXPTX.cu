#include  "GPUOptixPTX.cuh"

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

// Meta Closest Hit Shader
template<class PGroup>
__device__ void KCClosestHit()
{
    using HitStruct = typename PGroup::HitData;
    using HitRecord = typename Record<typename PGroup::PrimitiveData,
                                      typename PGroup::LeafData>;
    const HitRecord* r = (const HitRecord*) optixGetSbtDataPointer();

    const int leafId = optixGetPrimitiveIndex();
    const uint32_t rayId = optixGetLaunchIndex().x;

    // Fetch the workKey, transformId, primitiveId from table
    PrimitiveId pId = r->gLeafs[leafId].primitiveId;
    TransformId tId = r->transformId;
    HitKey key = r->gLeafs[leafId].matId;

    // Write to the global memory
    params.gPrimitiveIds[rayId] = pId;
    params.gTransformIds[rayId] = tId;
    params.gWorkKeys[rayId] = key;

    // Read Attributes
    HitStruct s = {};
    uint32_t* hitStructAsInts = reinterpret_cast<uint32_t*>(&s);

    // TODO: Fix this

    constexpr uint32_t HitStructRegSize = ((sizeof(HitStruct) + sizeof(uint32_t) - 1)
                                            / sizeof(uint32_t));
    // This should get optimized out I hope
    // This is the most ghetto way I could have created i think
    if(0 <= HitStructRegSize) hitStructAsInts[0] = optixGetAttribute_0();
    if(1 <= HitStructRegSize) hitStructAsInts[1] = optixGetAttribute_1();
    if(2 <= HitStructRegSize) hitStructAsInts[2] = optixGetAttribute_2();
    if(3 <= HitStructRegSize) hitStructAsInts[3] = optixGetAttribute_3();
    if(4 <= HitStructRegSize) hitStructAsInts[4] = optixGetAttribute_4();
    if(5 <= HitStructRegSize) hitStructAsInts[5] = optixGetAttribute_5();
    if(6 <= HitStructRegSize) hitStructAsInts[6] = optixGetAttribute_6();
    if(7 <= HitStructRegSize) hitStructAsInts[7] = optixGetAttribute_7();

    // TODO: Our original barycentrics was wrong?
    // Change that or this
    if constexpr(std::is_same_v<PGroup, GPUPrimitiveTriangle>)
    {
        // MRay barycentric order is different
        float c = 1 - s[0] - s[1];
        s = Vector2f(c, s[0]);
    }

    // Finally write the hit struct as well
    params.gHitStructs.Ref<HitStruct>(rayId) = s;
}

// Meta Any Hit Shader
template<class PGroup>
__device__ void KCAnyHit()
{
    //using HitStruct = typename PGroup::HitData;
    //using Record = typename HitGroupRecord<typename PGroup::PrimitiveData,
    //                                       typename PGroup::LeafData>::Data;
    //const Record& r = *(const Record*)optixGetSbtDataPointer();

    //const int leafId = optixGetPrimitiveIndex();

    // Call Alpha Check
    //float newT;

    //bool missed = PGroup::AlphaTest();

    //if(missed) optixIgnoreIntersection();
}

// Meta Intersect Shader
template<class PGroup>
__device__ void KCIntersect()
{
    //using HitStruct = typename PGroup::HitData;
    //using Record = typename HitGroupRecord<typename PGroup::PrimitiveData,
    //                                       typename PGroup::LeafData>::Data;
    //const Record& r = *(const Record*)optixGetSbtDataPointer();

    //const int leafId = optixGetPrimitiveIndex();

    //// Do Intersect

    ////
    ////optixReportIntersection(
    ////    t * l,
    ////    HIT_OUTSIDE_FROM_INSIDE,
    ////    float3_as_args(normal));
}

__device__
void KCMissOptiX()
{
    // Do Nothing
}

__device__
void KCRayGenOptix()
{
    // We Launch linearly
    const uint32_t theLaunchDim = optixGetLaunchDimensions().x;
    const uint32_t theLaunchIndex = optixGetLaunchIndex().x;
    // Load Ray
    RayReg ray(params.gRays, theLaunchIndex);

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
               0, 1, 0);
}

// Actual Definitions
WRAP_FUCTION(__raygen__OptiX, KCRayGenOptix);
WRAP_FUCTION(__miss__OptiX, KCMissOptiX);
// Using this style instead of demangling function names etc.
// Triangle
WRAP_FUCTION(__closesthit__Triangle, KCClosestHit<GPUPrimitiveTriangle>)
WRAP_FUCTION(__anyhit__Triangle, KCAnyHit<GPUPrimitiveTriangle>)
// Sphere
WRAP_FUCTION(__closesthit__Sphere, KCClosestHit<GPUPrimitiveSphere>)
WRAP_FUCTION(__anyhit__Sphere, KCAnyHit<GPUPrimitiveSphere>)
WRAP_FUCTION(__intersection__Sphere, KCIntersect<GPUPrimitiveSphere>)