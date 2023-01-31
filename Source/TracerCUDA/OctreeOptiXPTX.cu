#include  "OctreeOptiXPTX.cuh"
#include "RayLib/RandomColor.h"

extern "C" __constant__ OctreeAccelParams params;

// Meta Closest Hit Shader
__device__ __forceinline__
void KCClosestHitSVO()
{
    const int leafId = optixGetPrimitiveIndex();
    optixSetPayload_1(leafId);
    optixSetPayload_2(__float_as_uint(optixGetRayTmax()));
}

__device__ __forceinline__
void KCMissSVOOptiX()
{
    // Do Nothing
}

template <class T>
__device__ __forceinline__
void KCIntersectVoxel()
{
    // TODO: Docs says object space is faster?
    // check it
    float3 rayOrig = optixGetWorldRayOrigin();
    float3 rayDir = optixGetWorldRayDirection();
    uint32_t currentLevel = optixGetPayload_0();

    const int leafId = optixGetPrimitiveIndex();
    const T* dMortonCodes = reinterpret_cast<const T*>(optixGetSbtDataPointer());

    // Although AABB == the Voxel, we cant query the hit tMin
    // from the API (probably it does not have it)
    // Do the intersection "by hand"
    RayF ray = RayF(Vector3f(rayDir.x,
                             rayDir.y,
                             rayDir.z),
                    Vector3f(rayOrig.x,
                             rayOrig.y,
                             rayOrig.z));
    Vector2f tMinMax = Vector2f(optixGetRayTmin(),
                                optixGetRayTmax());
    // Generate the AABB
    // "Decompress the Morton code to AABB"
    AABB3f octreeAABB = params.svo.OctreeAABB();
    float levelVoxSize = params.svo.LevelVoxelSize(currentLevel);
    Vector3ui voxId = MortonCode::Decompose3D<T>(dMortonCodes[leafId]);
    Vector3f voxIdF = Vector3f(voxId);
    Vector3f voxAABBMin = octreeAABB.Min() + voxIdF * levelVoxSize;
    Vector3f voxAABBMax = voxAABBMin + levelVoxSize;
    // Actual AABB intersection
    float newT;
    Vector3f position;
    if(ray.IntersectsAABB(position, newT,
                          voxAABBMin,
                          voxAABBMax,
                          tMinMax))
    {
        optixReportIntersection(newT, 0);
    }
}

__device__ __forceinline__
void KCCamTraceSVO()
{
    // We Launch linearly
    const uint32_t launchDim = optixGetLaunchDimensions().x;
    const uint32_t launchIndex = optixGetLaunchIndex().x;
    // Should we check this ??
    if(launchIndex >= launchDim) return;

    // TODO: How do we generate rays
    RayReg ray = RayReg(params.gRays, launchIndex);

    uint32_t foundLevel;
    uint32_t nodeRelativeIndex;


    float coneAperture = 0.0f;
    const AnisoSVOctreeGPU& svo = params.svo;
    for(int i = svo.LeafDepth(); i > 0; i--)
    {
        // Local Parameters
        // Which node did we hit?
        // tMin of the Hit
        uint32_t localHitNodeIndex = 0;
        uint32_t tMinOutUInt32 = __float_as_uint(FLT_MAX);
        uint32_t currentLevel = static_cast<uint32_t>(i);


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
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   // SBT
                   i, 1, 0,
                   currentLevel,
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
    uint32_t globalNodeId = svo.LevelNodeStart(foundLevel) + nodeRelativeIndex;
    float radiance = svo.ReadRadiance(ray.ray.getDirection(),
                                      coneAperture,
                                      globalNodeId, isLeaf);

    // Write to heap (...)
    uint32_t sampleIndex = params.gRayAux[launchIndex].sampleIndex;

    // Octree Display Mode
    Vector3f rayDir = ray.ray.getDirection();
    Vector4f locColor = Vector4f(0.0f, 0.0f, 10.0f, 1.0f);
    WFPGRenderMode mode = params.renderMode;
    if(mode == WFPGRenderMode::SVO_FALSE_COLOR)
        locColor = (globalNodeId != UINT32_MAX) ? Vector4f(Utility::RandomColorRGB(globalNodeId), 1.0f)
        : Vector4f(Vector3f(0.0f), 1.0f);
    // Payload Display Mode
    else if(globalNodeId == UINT32_MAX)
        locColor = Vector4f(1.0f, 0.0f, 1.0f, 1.0f);
    else if(mode == WFPGRenderMode::SVO_RADIANCE)
    {
        //Vector3f hitPos = ray.ray.getPosition() + rayDir.Normalize() * tMin;
        //float radianceF = ReadInterpolatedRadiance(hitPos, rayDir,
        //                                           params.pixelAperture,
        //                                           svo);

        half radiance = svo.ReadRadiance(rayDir, params.pixelAperture,
                                         globalNodeId, isLeaf);
        float radianceF = radiance;
        //if(radiance != static_cast<half>(MRAY_HALF_MAX))
        locColor = Vector4f(Vector3f(radianceF), 1.0f);
    }
    else if(mode == WFPGRenderMode::SVO_NORMAL)
    {
        float stdDev;
        Vector3f normal = svo.DebugReadNormal(stdDev, globalNodeId, isLeaf);

        // Voxels are two sided show the normal for the current direction
        normal = (normal.Dot(rayDir) >= 0.0f) ? normal : -normal;

        // Convert normal to 0-1 range
        normal += Vector3f(1.0f);
        normal *= Vector3f(0.5f);
        locColor = Vector4f(normal, stdDev);
    }

    // Actual Write
    params.gSamples.gValues[sampleIndex] = locColor;
}

__device__ __forceinline__
void KCRadGenSVO()
{

}

WRAP_FUCTION(__raygen__SVOCamTrace, KCCamTraceSVO);
WRAP_FUCTION(__raygen__SVORadGen, KCRadGenSVO);
WRAP_FUCTION(__miss__SVO, KCMissSVOOptiX);
WRAP_FUCTION(__closesthit__SVO, KCClosestHitSVO);
WRAP_FUCTION(__intersection__SVOMorton32, KCIntersectVoxel<uint32_t>)
WRAP_FUCTION(__intersection__SVOMorton64, KCIntersectVoxel<uint64_t>)