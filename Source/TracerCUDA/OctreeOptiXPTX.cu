#include  "OctreeOptiXPTX.cuh"

extern "C" __constant__ OctreeAccelParams params;

// Meta Closest Hit Shader
__device__ void KCClosestHitSVO()
{
    const int leafId = optixGetPrimitiveIndex();
    optixSetPayload_0(leafId);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

__device__
void KCMissSVOOptiX()
{
    // Do Nothing
}

__device__
void KCOCtreeConeTraceOptix()
{
    // We Launch linearly
    const uint32_t launchDim = optixGetLaunchDimensions().x;
    const uint32_t launchIndex = optixGetLaunchIndex().x;

    // TODO: How do we generate rays
    RayReg ray;// (params.gRays, launchIndex);


    uint32_t foundLevel;
    uint32_t nodeRelativeIndex;

    const AnisoSVOctreeGPU& svo = params.svo;
    float coneAperture = 0.0f;


    for(int i = svo.LeafDepth(); i > 0; i--)
    {
        uint32_t localHitNodeIndex = 0;



        uint32_t tMinOutUInt32 = __float_as_uint(FLT_MAX);
        optixTrace(// Accelerator
                   params.octreeLevelBVHs[i],
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
                   OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                   // SBT
                   0, 1, 0,
                   localHitNodeIndex,
                   tMinOutUInt32);


        static_assert(sizeof(uint32_t) == sizeof(float));
        float tMinOut = __uint_as_float(tMinOutUInt32);

        // If hit is found (tMinOut != FLT_MAX)
        // Check the cone radius
        // If cone radius and etc aligned with parameters and
        // tMin < tminCurrent
        // accept the hit
        foundLevel = i;
    }

    // Finally Query the result
    bool isLeaf = (foundLevel == svo.LeafDepth());
    uint32_t globalNodeId = /*svo.LevelNodeStart() +*/ nodeRelativeIndex;
    float radiance = svo.ReadRadiance(ray.ray.getDirection(),
                                      coneAperture,
                                      globalNodeId, isLeaf);

    // Write to heap (...)

}

//WRAP_FUCTION(__raygen__SVO, KCOCtreeCamTraceOptix);
WRAP_FUCTION(__raygen__SVO, KCOCtreeConeTraceOptix);
WRAP_FUCTION(__miss__SVO, KCMissSVOOptiX);
WRAP_FUCTION(__closesthit__SVO, KCClosestHitSVO);

//WRAP_FUCTION(__intersection__SVO, KCIntersect<GPUPrimitiveSphere>)