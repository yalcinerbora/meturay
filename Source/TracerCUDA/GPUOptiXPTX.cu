#include  "GPUOptiXPTX.cuh"

#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"
#include "GPUTransformEmpty.cuh"

struct RayPayload
{
    HitKey*         gRayWorkKey;
    TransformId*    gRayTransformId;
    PrimitiveId*    gRayPrimitiveId;
    HitStructPtr    gRayHitStruct;
};

extern "C" __constant__ OpitXBaseAccelParams params;

template <class HitStruct>
__device__ __forceinline__
HitStruct ReadHitStructFromAttribs()
{
    constexpr uint32_t HitStructRegSize = ((sizeof(HitStruct) + sizeof(uint32_t) - 1)
                                           / sizeof(uint32_t));

    // TODO: Maybe Implement this better without any UB
    //const uint32_t* hitStructAsInts = reinterpret_cast<const uint32_t*>(&hs);

    // Non UB version
    uint32_t hitStructAsInts[HitStructRegSize];
    // This should get optimized out I hope
    // This is the most ghetto way I could have created i think
    static_assert(HitStructRegSize <= 8, "Hit Struct is too large for OptiX");
    if(1 <= HitStructRegSize) hitStructAsInts[0] = optixGetAttribute_0();
    if(2 <= HitStructRegSize) hitStructAsInts[1] = optixGetAttribute_1();
    if(3 <= HitStructRegSize) hitStructAsInts[2] = optixGetAttribute_2();
    if(4 <= HitStructRegSize) hitStructAsInts[3] = optixGetAttribute_3();
    if(5 <= HitStructRegSize) hitStructAsInts[4] = optixGetAttribute_4();
    if(6 <= HitStructRegSize) hitStructAsInts[5] = optixGetAttribute_5();
    if(7 <= HitStructRegSize) hitStructAsInts[6] = optixGetAttribute_6();
    if(8 <= HitStructRegSize) hitStructAsInts[7] = optixGetAttribute_7();

    // Non UB version
    HitStruct s = {};
    memcpy(&s, reinterpret_cast<const Byte*>(hitStructAsInts), sizeof(HitStruct));
    return s;
}

template <class HitStruct>
__device__ __forceinline__
void ReportIntersection(float newT, unsigned int kind,
                        const HitStruct hs)
{
    constexpr uint32_t HitStructRegSize = ((sizeof(HitStruct) + sizeof(uint32_t) - 1)
                                           / sizeof(uint32_t));

    // Pre-check the Empty (C++ sizeof empty struct is 1
    // so this should never be branched)
    // But device maybe it is different ??
    if constexpr(0 == HitStructRegSize)
    {
        optixReportIntersection(newT, kind);
        return;
    }

    // TODO: Maybe Implement this better without any UB
    //const uint32_t* hitStructAsInts = reinterpret_cast<const uint32_t*>(&hs);
    // Non UB version
    uint32_t hitStructAsInts[HitStructRegSize];
    memcpy(hitStructAsInts, reinterpret_cast<const Byte*>(&hs), sizeof(HitStruct));

    // You think above code was bas, watch this
    // Instead of switch I am using constexpr if
    // it hints that only one of this functions will be
    // available in this function better I think.
    if constexpr(1 == HitStructRegSize)
        optixReportIntersection(newT, kind,
                                hitStructAsInts[0]);
    else if constexpr(2 == HitStructRegSize)
        optixReportIntersection(newT, kind,
                                hitStructAsInts[0],
                                hitStructAsInts[1]);
    else if constexpr(3 == HitStructRegSize)
        optixReportIntersection(newT, kind,
                                hitStructAsInts[0],
                                hitStructAsInts[1],
                                hitStructAsInts[2]);
    else if constexpr(4 == HitStructRegSize)
        optixReportIntersection(newT, kind,
                                hitStructAsInts[0],
                                hitStructAsInts[1],
                                hitStructAsInts[2],
                                hitStructAsInts[3]);
    else if constexpr(5 == HitStructRegSize)
        optixReportIntersection(newT, kind,
                                hitStructAsInts[0],
                                hitStructAsInts[1],
                                hitStructAsInts[2],
                                hitStructAsInts[3],
                                hitStructAsInts[4]);
    else if constexpr(6 == HitStructRegSize)
        optixReportIntersection(newT, kind,
                                hitStructAsInts[0],
                                hitStructAsInts[1],
                                hitStructAsInts[2],
                                hitStructAsInts[3],
                                hitStructAsInts[4],
                                hitStructAsInts[5]);
    else if constexpr(7 == HitStructRegSize)
        optixReportIntersection(newT, kind,
                                hitStructAsInts[0],
                                hitStructAsInts[1],
                                hitStructAsInts[2],
                                hitStructAsInts[3],
                                hitStructAsInts[4],
                                hitStructAsInts[5],
                                hitStructAsInts[6]);
    else if constexpr(8 == HitStructRegSize)
        optixReportIntersection(newT, kind,
                                hitStructAsInts[0],
                                hitStructAsInts[1],
                                hitStructAsInts[2],
                                hitStructAsInts[3],
                                hitStructAsInts[4],
                                hitStructAsInts[5],
                                hitStructAsInts[6],
                                hitStructAsInts[7]);

    static_assert(HitStructRegSize <= 8, "Hit Struct is too large for OptiX");
}

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
    HitStruct hitStruct = ReadHitStructFromAttribs<HitStruct>();

    // TODO: Our original barycentrics was wrong?
    // Change that or this
    if constexpr(std::is_same_v<PGroup, GPUPrimitiveTriangle>)
    {
        // MRay barycentric order is different
        float c = 1 - hitStruct[0] - hitStruct[1];
        hitStruct = Vector2f(c, hitStruct[0]);
    }

    // Finally write the hit struct as well
    params.gHitStructs.Ref<HitStruct>(rayId) = hitStruct;
}

// Meta Any Hit Shader
template<class PGroup>
__device__ void KCAnyHit()
{
    using LeafStruct = typename PGroup::LeafData;
    using HitStruct  = typename PGroup::HitData;
    using HitRecord  = typename Record<typename PGroup::PrimitiveData,
                                      typename PGroup::LeafData>;
    // Record Fetch
    const HitRecord* r = (const HitRecord*)optixGetSbtDataPointer();
    const int leafId = optixGetPrimitiveIndex();
    // Fetch Leaf and Hit
    HitStruct potentialHit = ReadHitStructFromAttribs<HitStruct>();
    const LeafStruct& gLeaf = r->gLeafs[leafId];

    bool passed = PGroup::AlphaTest(potentialHit, gLeaf, (*r->gPrimData));
    if(!passed) optixIgnoreIntersection();
}

// Meta Intersect Shader
template<class PGroup>
__device__ void KCIntersect()
{
    //GPUTransformIdentity ID_TRANSFORM;

    using LeafStruct = typename PGroup::LeafData;
    using HitStruct  = typename PGroup::HitData;
    using HitRecord  = typename Record<typename PGroup::PrimitiveData,
                                       typename PGroup::LeafData>;
    // Construct a ray
    float3 rP = optixGetWorldRayOrigin();
    float3 rD = optixGetWorldRayDirection();
    const float  tMin = optixGetRayTmin();
    const float  tMax = optixGetRayTmax();

    // Record Fetch
    const HitRecord* r = (const HitRecord*)optixGetSbtDataPointer();
    const int leafId = optixGetPrimitiveIndex();
    // Fetch Leaf
    const LeafStruct& gLeaf = r->gLeafs[leafId];

    // Outputs
    float newT;
    HitStruct hitData;
    bool intersects = false;
    // Our intersects function requires a transform
    // Why?
    //
    // For skinned meshes (or any primitive with a transform that cannot be applied
    // to a ray inversely) each accelerator is fully constructed using transformed aabbs,
    //
    // However primitives are not transformed, thus for each shader we need the transform
    if constexpr(PGroup::TransType == PrimTransformType::CONSTANT_LOCAL_TRANSFORM)
    {
        // Use optix transform the ray to local (object) space of the primitive
        // and use an identity transform on the Intersection function
        rD = optixTransformVectorFromWorldToObjectSpace(rD);
        rP = optixTransformPointFromWorldToObjectSpace(rP);
    }
    else if constexpr(PGroup::TransType == PrimTransformType::PER_PRIMITIVE_TRANSFORM)
    {
        static_assert(false, "Per primitive transform is not supported on OptiX yet");
        // Optix does not support virtual function calls
        // Just leave it as is for now
    }
    else static_assert(false, "Primitive does not have proper transform type");

    // Construct the Register (after transformation)
    RayReg rayRegister = RayReg(RayF(Vector3f(rD.x, rD.y, rD.z),
                                     Vector3f(rP.x, rP.y, rP.z)),
                                tMin, tMax);

    // Since OptiX does not support virtual(indirect) function calls
    // Call with an empty transform (this is not virutal and does nothing)
    intersects = PGroup::IntersectsT<GPUTransformEmpty>(// Output
                                                        newT,
                                                        hitData,
                                                        // I-O
                                                        rayRegister,
                                                        // Input
                                                        GPUTransformEmpty(),
                                                        gLeaf,
                                                        (*r->gPrimData));
    // Report the intersection
    if(intersects) ReportIntersection<HitStruct>(newT, 0, hitData);
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